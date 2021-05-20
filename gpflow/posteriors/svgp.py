#  Copyright 2021 The GPflow Contributors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from abc import ABC
from typing import Optional

import tensorflow as tf

from .. import covariances, kernels, mean_functions
from ..conditionals.util import (
    base_conditional,
    expand_independent_outputs,
    fully_correlated_conditional,
    independent_interdomain_conditional,
    mix_latent_gp,
    separate_independent_conditional_implementation,
)
from ..config import default_jitter
from ..inducing_variables import SharedIndependentInducingVariables
from ..types import MeanAndVariance
from .posterior import (
    Posterior,
    PrecomputeCacheType,
    VariationalPosteriorMixin,
    _DeltaDist,
    _DiagNormal,
    _MvNormal,
)


class BasePosterior(Posterior, VariationalPosteriorMixin, ABC):
    def __init__(
        self,
        kernel,
        inducing_variable,
        q_mu: tf.Tensor,
        q_sqrt: tf.Tensor,
        whiten: bool = True,
        mean_function: Optional[mean_functions.MeanFunction] = None,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ):
        """
        Users should use `create_posterior` to create instances of concrete
        subclasses of this Posterior class instead of calling this constructor
        directly. For `create_posterior` to be able to correctly instantiate
        subclasses, developers need to ensure their subclasses don't change the
        constructor signature.
        """
        self.inducing_variable = inducing_variable
        self.whiten = whiten
        self._set_qdist(q_mu, q_sqrt)

        super().__init__(kernel, mean_function, precompute_cache=precompute_cache)

    def _precompute(self):
        Kuu = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [(R), M, M]
        q_mu = self._q_dist.q_mu

        if Kuu.shape.ndims == 4:
            ML = tf.reduce_prod(tf.shape(Kuu)[:2])
            Kuu = tf.reshape(Kuu, [ML, ML])
        if Kuu.shape.ndims == 3:
            q_mu = tf.linalg.adjoint(self._q_dist.q_mu)[..., None]  # [..., R, M, 1]
        L = tf.linalg.cholesky(Kuu)

        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = tf.linalg.cholesky_solve(L, q_mu)
        else:
            # alpha = L⁻ᵀ q_mu
            alpha = tf.linalg.triangular_solve(L, q_mu, adjoint=True)
        # predictive mean = Kfu alpha
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtᵀ
        I = tf.eye(tf.shape(L)[-1], dtype=L.dtype)
        if isinstance(self._q_dist, _DeltaDist):
            B = I
        else:
            if not self.whiten:
                # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
                #      = Kuu⁻¹ - L⁻ᵀ L⁻¹ S L⁻ᵀ L⁻¹
                #      = L⁻ᵀ (I - L⁻¹ S L⁻ᵀ) L⁻¹
                #      = L⁻ᵀ B L⁻¹
                if isinstance(self._q_dist, _DiagNormal):
                    q_sqrt = tf.linalg.diag(tf.linalg.adjoint(self._q_dist.q_sqrt))
                elif isinstance(self._q_dist, _MvNormal):
                    q_sqrt = self._q_dist.q_sqrt
                Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)
                Linv_cov_u_LinvT = tf.matmul(Linv_qsqrt, Linv_qsqrt, transpose_b=True)
            else:
                if isinstance(self._q_dist, _DiagNormal):
                    Linv_cov_u_LinvT = tf.linalg.diag(tf.linalg.adjoint(self._q_dist.q_sqrt ** 2))
                elif isinstance(self._q_dist, _MvNormal):
                    q_sqrt = self._q_dist.q_sqrt
                    Linv_cov_u_LinvT = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
                # Qinv = Kuu⁻¹ - L⁻ᵀ S L⁻¹
                # Linv = (L⁻¹ I) = solve(L, I)
                # Kinv = Linvᵀ @ Linv
            B = I - Linv_cov_u_LinvT
        LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        B_Linv = tf.linalg.adjoint(LinvT_B)
        Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)

        M, L = tf.unstack(tf.shape(self._q_dist.q_mu), num=2)
        Qinv = tf.broadcast_to(Qinv, [L, M, M])

        tf.debugging.assert_shapes(
            [(Qinv, ["L", "M", "M"]),]
        )

        return alpha, Qinv


class IndependentPosterior(BasePosterior):
    def _post_process_mean_and_cov(self, mean, cov, full_cov, full_output_cov):
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def _get_Kff(self, Xnew, full_cov):

        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        if isinstance(self.kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would return
            # if full_cov: [P, N, N] -- this is what we want
            # else: [N, P] instead of [P, N] as we get from the explicit stack below
            Kff = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, kernels.MultioutputKernel):
            # effectively, SharedIndependent path
            Kff = self.kernel.kernel(Xnew, full_cov=full_cov)
            # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would return
            # if full_cov: [P, N, N] instead of [N, N]
            # else: [N, P] instead of [N]
        else:
            # standard ("single-output") kernels
            Kff = self.kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

        return Kff

    def _conditional_with_precompute(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # Qinv: [L, M, M]
        # alpha: [M, L]

        Kuf = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [(R), M, N]
        Kff = self._get_Kff(Xnew, full_cov)

        mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if full_cov:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, self.Qinv @ Kuf, transpose_a=True)
            cov = Kff - Kfu_Qinv_Kuf
        else:
            # [Aᵀ B]_ij = Aᵀ_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(self.Qinv, Kuf), axis=-2)
            cov = Kff - Kfu_Qinv_Kuf
            cov = tf.linalg.adjoint(cov)

        return self._post_process_mean_and_cov(mean, cov, full_cov, full_output_cov)


class IndependentPosteriorSingleOutput(IndependentPosterior):
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # same as IndependentPosteriorMultiOutput, Shared~/Shared~ branch, except for following line:
        Knn = self.kernel(Xnew, full_cov=full_cov)

        Kmm = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [M, M]
        Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        fmean, fvar = base_conditional(
            Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
        )  # [N, P],  [P, N, N] or [N, P]
        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class IndependentPosteriorMultiOutput(IndependentPosterior):
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        if isinstance(self.inducing_variable, SharedIndependentInducingVariables) and isinstance(
            self.kernel, kernels.SharedIndependent
        ):
            # same as IndependentPosteriorSingleOutput except for following line
            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
            # we don't call self.kernel() directly as that would do unnecessary tiling

            Kmm = covariances.Kuu(
                self.inducing_variable, self.kernel, jitter=default_jitter()
            )  # [M, M]
            Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [N, P],  [P, N, N] or [N, P]
        else:
            # this is the messy thing with tf.map_fn, cleaned up by the st/clean_up_broadcasting_conditionals branch

            # Following are: [P, M, M]  -  [P, M, N]  -  [P, N](x N)
            Kmms = covariances.Kuu(
                self.inducing_variable, self.kernel, jitter=default_jitter()
            )  # [P, M, M]
            Kmns = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [P, M, N]
            if isinstance(self.kernel, kernels.Combination):
                kernel_list = self.kernel.kernels
            else:
                kernel_list = [self.kernel.kernel] * len(
                    self.inducing_variable.inducing_variable_list
                )
            Knns = tf.stack(
                [k.K(Xnew) if full_cov else k.K_diag(Xnew) for k in kernel_list], axis=0
            )

            fmean, fvar = separate_independent_conditional_implementation(
                Kmns,
                Kmms,
                Knns,
                self.q_mu,
                q_sqrt=self.q_sqrt,
                full_cov=full_cov,
                white=self.whiten,
            )

        return self._post_process_mean_and_cov(fmean, fvar, full_cov, full_output_cov)


class LinearCoregionalizationPosterior(IndependentPosteriorMultiOutput):
    def _post_process_mean_and_cov(self, mean, cov, full_cov, full_output_cov):
        """
        mean: [N, L]
        cov: [L, N, N] or [N, L]
        """
        cov = expand_independent_outputs(cov, full_cov, full_output_cov=False)
        mean, cov = mix_latent_gp(self.kernel.W, mean, cov, full_cov, full_output_cov)
        return mean, cov


class FullyCorrelatedPosterior(BasePosterior):
    def _conditional_with_precompute(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        # Qinv: [L, M, M]
        # alpha: [M, L]

        Kuf = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        assert Kuf.shape.ndims == 4
        M, L, N, K = tf.unstack(tf.shape(Kuf), num=Kuf.shape.ndims, axis=0)
        Kuf = tf.reshape(Kuf, (M * L, N * K))

        Kff = self.kernel(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # full_cov=True and full_output_cov=True: [N, P, N, P]
        # full_cov=True and full_output_cov=False: [P, N, N]
        # full_cov=False and full_output_cov=True: [N, P, P]
        # full_cov=False and full_output_cov=False: [N, P]
        if full_cov == full_output_cov:
            new_shape = (N * K, N * K) if full_cov else (N * K,)
            Kff = tf.reshape(Kff, new_shape)

        N = tf.shape(Xnew)[0]
        K = tf.shape(Kuf)[-1] // N

        mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if not full_cov and not full_output_cov:
            # fully diagonal case in both inputs and outputs
            # [Aᵀ B]_ij = Aᵀ_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(self.Qinv, Kuf), axis=-2)
        else:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, self.Qinv @ Kuf, transpose_a=True)
            if not (full_cov and full_output_cov):
                # diagonal in either inputs or outputs
                new_shape = tf.concat([tf.shape(Kfu_Qinv_Kuf)[:-2], (N, K, N, K)], axis=0)
                Kfu_Qinv_Kuf = tf.reshape(Kfu_Qinv_Kuf, new_shape)
                if full_cov:
                    # diagonal in outputs: move outputs to end
                    tmp = tf.linalg.diag_part(tf.einsum("...ijkl->...ikjl", Kfu_Qinv_Kuf))
                elif full_output_cov:
                    # diagonal in inputs: move inputs to end
                    tmp = tf.linalg.diag_part(tf.einsum("...ijkl->...jlik", Kfu_Qinv_Kuf))
                Kfu_Qinv_Kuf = tf.einsum("...ijk->...kij", tmp)  # move diagonal dim to [-3]
        cov = Kff - Kfu_Qinv_Kuf

        if not full_cov and not full_output_cov:
            cov = tf.linalg.adjoint(cov)

        mean = tf.reshape(mean, (N, K))
        if full_cov == full_output_cov:
            cov_shape = (N, K, N, K) if full_cov else (N, K)
        else:
            cov_shape = (K, N, N) if full_cov else (N, K, K)
        cov = tf.reshape(cov, cov_shape)

        return mean, cov

    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        Kmm = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [M, L, M, L]
        Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, L, N, P]
        Knn = self.kernel(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )  # [N, P](x N)x P  or  [N, P](x P)

        M, L, N, K = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Kmm = tf.reshape(Kmm, (M * L, M * L))

        if full_cov == full_output_cov:
            Kmn = tf.reshape(Kmn, (M * L, N * K))
            Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
            mean, cov = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [K, 1], [1, K](x NK)
            mean = tf.reshape(mean, (N, K))
            cov = tf.reshape(cov, (N, K, N, K) if full_cov else (N, K))
        else:
            Kmn = tf.reshape(Kmn, (M * L, N, K))
            mean, cov = fully_correlated_conditional(
                Kmn,
                Kmm,
                Knn,
                self.q_mu,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=self.q_sqrt,
                white=self.whiten,
            )
        return mean, cov


class FallbackIndependentLatentPosterior(FullyCorrelatedPosterior):  # XXX
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        Kmm = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [L, M, M]
        Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, L, N, P]
        Knn = self.kernel(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )  # [N, P](x N)x P  or  [N, P](x P)

        return independent_interdomain_conditional(
            Kmn,
            Kmm,
            Knn,
            self.q_mu,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
            q_sqrt=self.q_sqrt,
            white=self.whiten,
        )
