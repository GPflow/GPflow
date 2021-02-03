# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import covariances, kernels
from .base import Module, Parameter
from .conditionals.util import (
    base_conditional,
    expand_independent_outputs,
    fully_correlated_conditional,
    independent_interdomain_conditional,
    mix_latent_gp,
    separate_independent_conditional_implementation,
)
from .config import default_float, default_jitter
from .inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    InducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from .types import MeanAndVariance
from .utilities import Dispatcher


class DeltaDist(Module):
    def __init__(self, q_mu):
        self.q_mu = q_mu  # [M, L]

    @property
    def q_sqrt(self):
        return None


class DiagNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [M, L]


class MvnNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [L, M, M], lower-triangular


class AbstractPosterior(Module, ABC):
    def __init__(
        self,
        kernel,
        inducing_variable,
        q_mu,
        q_sqrt,
        whiten=True,
        mean_function=None,
        precompute=True,
    ):
        self.inducing_variable = inducing_variable
        self.kernel = kernel
        self.mean_function = mean_function
        self.whiten = whiten
        self._set_qdist(q_mu, q_sqrt)

        if precompute:
            self.update_cache()  # populates or updates self.alpha and self.Qinv
        # NOTE we CANNOT use update_cache_with_variables() here,
        # as that would break gradient flow in training

    @property
    def q_mu(self):
        return self.q_dist.q_mu

    @property
    def q_sqrt(self):
        return self.q_dist.q_sqrt

    def _set_qdist(self, q_mu, q_sqrt):
        if q_sqrt is None:
            self.q_dist = DeltaDist(q_mu)
        elif len(q_sqrt.shape) == 2:  # q_diag
            self.q_dist = DiagNormal(q_mu, q_sqrt)
        else:
            self.q_dist = MvnNormal(q_mu, q_sqrt)

    def update_cache(self):
        self.alpha, self.Qinv = self._precompute()

    def freeze(self):
        alpha, Qinv = self._precompute()
        self.alpha = Parameter(alpha, trainable=False)
        self.Qinv = Parameter(Qinv, trainable=False)

    def update_cache_with_variables(self):
        alpha, Qinv = self._precompute()
        if isinstance(self.alpha, Parameter) and isinstance(self.Qinv, Parameter):
            self.alpha.assign(alpha)
            self.Qinv.assign(Qinv)
        else:
            self.alpha = Parameter(alpha, trainable=False)
            self.Qinv = Parameter(Qinv, trainable=False)

    @abstractmethod
    def _precompute(self) -> Tuple[tf.Tensor]:
        """
        Precomputes alpha and Qinv that do not depend on Xnew
        """

    @abstractmethod
    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew
        Relies on precomputed alpha and Qinv (see _precompute method)
        """

    @abstractmethod
    def fused_predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew
        Does not make use of caching
        """


class BasePosterior(AbstractPosterior):
    def _precompute(self):
        Kuu = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [(R), M, M]
        q_mu = self.q_dist.q_mu

        if Kuu.shape.ndims == 4:
            ML = tf.reduce_prod(tf.shape(Kuu)[:2])
            Kuu = tf.reshape(Kuu, [ML, ML])
        if Kuu.shape.ndims == 3:
            q_mu = tf.linalg.adjoint(self.q_dist.q_mu)[..., None]  # [..., R, M, 1]
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
        if isinstance(self.q_dist, DeltaDist):
            B = I
        else:
            if not self.whiten:
                # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
                #      = Kuu⁻¹ - L⁻ᵀ L⁻¹ S L⁻ᵀ L⁻¹
                #      = L⁻ᵀ (I - L⁻¹ S L⁻ᵀ) L⁻¹
                #      = L⁻ᵀ B L⁻¹
                if isinstance(self.q_dist, DiagNormal):
                    q_sqrt = tf.linalg.diag(tf.linalg.adjoint(self.q_dist.q_sqrt))
                else:
                    q_sqrt = self.q_dist.q_sqrt
                Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)
                Linv_cov_u_LinvT = tf.matmul(Linv_qsqrt, Linv_qsqrt, transpose_b=True)
            else:
                if isinstance(self.q_dist, DiagNormal):
                    Linv_cov_u_LinvT = tf.linalg.diag(tf.linalg.adjoint(self.q_dist.q_sqrt ** 2))
                else:
                    q_sqrt = self.q_dist.q_sqrt
                    Linv_cov_u_LinvT = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
                # Qinv = Kuu⁻¹ - L⁻ᵀ S L⁻¹
                # Linv = (L⁻¹ I) = solve(L, I)
                # Kinv = Linvᵀ @ Linv
            B = I - Linv_cov_u_LinvT
        LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        B_Linv = tf.linalg.adjoint(LinvT_B)
        Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)

        return alpha, Qinv


class LinearOperatorBasePosterior(BasePosterior):
    # WIP

    # base Kuu [M, M]:
    #  - LinearOperatorFullMatrix
    #  - LinearOperatorBlockDiag
    #  - LinearOperatorDiag

    # Kernels / inducing variables:
    #  - single-output = shared IV *and* kernel; Kuu [M, M], Kuf [M, ..., N] / Kfu [..., N, M]
    #  - fully-correlated multi-output; Kuu [M, P, M, P], Kuf [M, P, ..., N, P] / Kfu [..., N, P, M, P]
    #  - separate independent IV and/or kernel (mainly for DGPs); Kuu [L=P, M, M], Kuf [L=P, ..., N] / Kfu [..., N, L=P, M]
    #  - LinearCoregionalization, IV in g-space; Kuu [L, M, M], Kuf [L, M, ..., N] / Kfu [..., N, L, M]
    #  - LinearCoregionalization, IV in f-space (fallback); Kuu [L, M, M], Kuf [M, L, ..., N, P] / Kfu [..., N, P, M, L]

    # q(u):
    #  - fully-correlated, [M*L, M*L]
    #    LinearOperatorFullMatrix(eye(M*L))
    #  - block-diagonal (mean field across L), [L, M, M]
    #    LinearOperatorBlockDiag([LinearOperatorFullMatrix(eye(M)) for _ in range(L)])
    #  - diagonal (mean field across L), [M, L] --> [L, M]?
    #    LinearOperatorDiag(ones(M*L))

    def _set_qdist(self, q_mu, q_sqrt):
        q_mu = tf.linalg.adjoint(q_mu)  # TODO redefine how q_mu is stored internally
        if len(q_sqrt.shape) == 2:  # q_diag
            q_sqrt = tf.linalg.adjoint(q_sqrt)  # TODO redefine how q_sqrt is stored internally
            tf.debugging.assert_shapes(
                [(q_mu, ["L", "M"]), (q_sqrt, ["L", "M"]),]
            )
            self.q_dist = tfp.distributions.MultivariateNormalDiag(loc=q_mu, scale_diag=q_sqrt)
        else:
            tf.debugging.assert_shapes(
                [(q_mu, ["L", "M"]), (q_sqrt, ["L", "M", "M"]),]
            )
            self.q_dist = tfp.distributions.MultivariateNormalTriL(loc=q_mu, scale_tril=q_sqrt)

    def _precompute(self):
        Kuu = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [(R), M, M]

        q_mu = self.q_dist.loc
        q_sqrt = self.q_dist.scale  # type: tf.linalg.LinearOperator

        if Kuu.shape.ndims == 4:
            # this branch only gets called in fully-correlated case
            # TODO this branch won't work; it's left-over from dense Kuu matrices
            ML = tf.reduce_prod(tf.shape(Kuu)[:2])
            Kuu = tf.reshape(Kuu, [ML, ML])

        L = Kuu.cholesky()

        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = L.solvevec(L.solvevec(q_mu), adjoint=True)  # type: tf.Tensor
        else:
            # alpha = L⁻ᵀ q_mu
            alpha = L.solvevec(q_mu, adjoint=True)  # type: tf.Tensor

        # predictive mean = Kfu alpha
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtᵀ
        if not self.whiten:
            # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
            #      = Kuu⁻¹ - L⁻ᵀ L⁻¹ S L⁻ᵀ L⁻¹
            #      = L⁻ᵀ (I - L⁻¹ S L⁻ᵀ) L⁻¹
            #      = L⁻ᵀ B L⁻¹
            # Linv_qsqrt = L.solve(q_sqrt)
            # Linv_cov_u_LinvT = Linv_qsqrt.matmul(Linv_qsqrt, adjoint_arg=True)
            KuuInv_qsqrt = L.solve(L.solve(q_sqrt), adjoint=True)
            KuuInv_covu_KuuInv = KuuInv_qsqrt.matmul(KuuInv_qsqrt, adjoint_arg=True)
        else:
            # Qinv = Kuu⁻¹ - L⁻ᵀ S L⁻¹
            # Linv = (L⁻¹ I) = solve(L, I)
            # Kinv = Linvᵀ @ Linv
            # Linv_cov_u_LinvT = q_sqrt.matmul(q_sqrt, adjoint_arg=True)
            LinvT_qsqrt = L.solve(q_sqrt, adjoint=True)
            KuuInv_covu_KuuInv = LinvT_qsqrt.matmul(LinvT_qsqrt, adjoint_arg=True)
        # I = tf.eye(tf.shape(Linv_cov_u_LinvT)[-1], dtype=Linv_cov_u_LinvT.dtype)
        # B = I - Linv_cov_u_LinvT
        # LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        # B_Linv = tf.linalg.adjoint(LinvT_B)
        # Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)

        Qinv = Kuu.inverse() - KuuInv_covu_KuuInv  # XXX LinearOperator does not support `-`

        return alpha, Qinv


class IndependentPosterior(BasePosterior):
    def _post_process_mean_and_cov(self, mean, cov, full_cov, full_output_cov):
        return mean, expand_independent_outputs(cov, full_cov, full_output_cov)

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # Qinv: [L, M, M]
        # alpha: [M, L]

        Kuf, Knn = _get_kernels(
            Xnew, self.inducing_variable, self.kernel, full_cov, full_output_cov
        )

        N = tf.shape(Xnew)[0]
        K = tf.shape(Kuf)[-1] // N

        mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if full_cov:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, self.Qinv @ Kuf, transpose_a=True)
            cov = Knn - Kfu_Qinv_Kuf
        else:
            # [Aᵀ B]_ij = Aᵀ_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(self.Qinv, Kuf), axis=-2)
            cov = Knn - Kfu_Qinv_Kuf
            cov = tf.linalg.adjoint(cov)

        mean, cov = self._post_process_mean_and_cov(mean, cov, full_cov, full_output_cov)
        return mean + self.mean_function(Xnew), cov


class IndependentPosteriorSingleOutput(IndependentPosterior):
    # could almost be the same as IndependentPosteriorMultiOutput ...
    def fused_predict_f(
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
    def fused_predict_f(
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
    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # TODO: this assumes that Xnew has shape [N, D] and no leading dims

        # Qinv: [L, M, M]
        # alpha: [M, L]

        Kuf = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        assert Kuf.shape.ndims == 4
        M, L, N, K = tf.unstack(tf.shape(Kuf), num=Kuf.shape.ndims, axis=0)
        Kuf = tf.reshape(Kuf, (M * L, N * K))

        Knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
        # full_cov=True and full_output_cov=True: [N, P, N, P]
        # full_cov=True and full_output_cov=False: [P, N, N]
        # full_cov=False and full_output_cov=True: [N, P, P]
        # full_cov=False and full_output_cov=False: [N, P]
        if full_cov == full_output_cov:
            new_shape = (N * K, N * K) if full_cov else (N * K,)
            Knn = tf.reshape(Knn, new_shape)

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
        cov = Knn - Kfu_Qinv_Kuf

        if not full_cov and not full_output_cov:
            cov = tf.linalg.adjoint(cov)

        mean = tf.reshape(mean, (N, K))
        if full_cov == full_output_cov:
            cov_shape = (N, K, N, K) if full_cov else (N, K)
        else:
            cov_shape = (K, N, N) if full_cov else (N, K, K)
        cov = tf.reshape(cov, cov_shape)

        return mean + self.mean_function(Xnew), cov

    def fused_predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        Kmm = covariances.Kuu(
            self.inducing_variable, self.kernel, jitter=default_jitter()
        )  # [M, L, M, L]
        Kmn = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, L, N, P]
        Knn = kernel(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )  # [N, P](x N)x P  or  [N, P](x P)

        M, L, N, K = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Kmm = tf.reshape(Kmm, (M * L, M * L))

        if full_cov == full_output_cov:
            Kmn = tf.reshape(Kmn, (M * L, N * K))
            Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
            fmean, fvar = base_conditional(
                Kmn, Kmm, Knn, self.q_mu, full_cov=full_cov, q_sqrt=self.q_sqrt, white=self.whiten
            )  # [K, 1], [1, K](x NK)
            fmean = tf.reshape(fmean, (N, K))
            fvar = tf.reshape(fvar, (N, K, N, K) if full_cov else (N, K))
        else:
            Kmn = tf.reshape(Kmn, (M * L, N, K))
            fmean, fvar = fully_correlated_conditional(
                Kmn,
                Kmm,
                Knn,
                self.q_mu,
                full_cov=full_cov,
                full_output_cov=full_output_cov,
                q_sqrt=self.q_sqrt,
                white=self.whiten,
            )
        return fmean, fvar


class FallbackIndependentLatentPosterior(FullyCorrelatedPosterior):  # XXX
    def fused_predict_f(
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


def _get_kernels(Xnew, inducing_variable, kernel, full_cov, full_output_cov):

    # TODO: this assumes that Xnew has shape [N, D] and no leading dims

    Kuf = covariances.Kuf(inducing_variable, kernel, Xnew)  # [(R), M, N]

    if isinstance(kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):
        # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would return
        # if full_cov: [P, N, N] -- this is what we want
        # else: [N, P] instead of [P, N] as we get from the explicit stack below
        Knn = tf.stack([k(Xnew, full_cov=full_cov) for k in kernel.kernels], axis=0)
    elif isinstance(kernel, kernels.MultioutputKernel):
        # effectively, SharedIndependent path
        Knn = kernel.kernel(Xnew, full_cov=full_cov)
        # NOTE calling kernel(Xnew, full_cov=full_cov, full_output_cov=False) directly would return
        # if full_cov: [P, N, N] instead of [N, N]
        # else: [N, P] instead of [N]
    else:
        # standard ("single-output") kernels
        Knn = kernel(Xnew, full_cov=full_cov)  # [N, N] if full_cov else [N]

    return Kuf, Knn


get_posterior_class = Dispatcher("get_posterior_class")


@get_posterior_class.register(kernels.Kernel, InducingVariables)
def _get_posterior_base_case(kernel, inducing_variable):
    # independent single output
    return IndependentPosteriorSingleOutput


@get_posterior_class.register(kernels.MultioutputKernel, InducingPoints)
def _get_posterior_fully_correlated_mo(kernel, inducing_variable):
    return FullyCorrelatedPosterior


@get_posterior_class.register(
    (kernels.SharedIndependent, kernels.SeparateIndependent),
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_independent_mo(kernel, inducing_variable):
    # independent multi-output
    return IndependentPosteriorMultiOutput


@get_posterior_class.register(
    kernels.IndependentLatent,
    (FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
)
def _get_posterior_independentlatent_mo_fallback(kernel, inducing_variable):
    return FallbackIndependentLatentPosterior


@get_posterior_class.register(
    kernels.LinearCoregionalization,
    (SeparateIndependentInducingVariables, SharedIndependentInducingVariables),
)
def _get_posterior_linearcoregionalization_mo_efficient(kernel, inducing_variable):
    # Linear mixing---efficient multi-output
    return LinearCoregionalizationPosterior


def create_posterior(kernel, inducing_variable, q_mu, q_sqrt, whiten=True, mean_function=None):
    posterior_class = get_posterior_class(kernel, inducing_variable)
    return posterior_class(kernel, inducing_variable, q_mu, q_sqrt, whiten, mean_function)
