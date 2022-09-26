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

from typing import NamedTuple, Optional, Tuple

import numpy as np
import tensorflow as tf

from .. import posteriors
from ..base import InputData, MeanAndVariance, RegressionData, TensorData
from ..config import default_float, default_jitter
from ..covariances.dispatch import Kuf, Kuu
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..likelihoods import Gaussian
from ..mean_functions import MeanFunction
from ..utilities import add_noise_cov, assert_params_false, to_default_float
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import InducingPointsLike, data_input_to_tensor, inducingpoint_wrapper


class SGPRBase_deprecated(GPModel, InternalDataTrainingLossMixin):
    """
    Common base class for SGPR and GPRFITC that provides the common __init__
    and upper_bound() methods.
    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
        "noise_variance: []",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPointsLike,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: Optional[TensorData] = None,
        likelihood: Optional[Gaussian] = None,
    ):
        """
        This method only works with a Gaussian likelihood, its variance is
        initialized to `noise_variance`.

        :param data: a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        :param inducing_variable:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        :param kernel: An appropriate GPflow kernel object.
        :param mean_function: An appropriate GPflow mean function object.
        """
        assert (noise_variance is None) or (
            likelihood is None
        ), "Cannot set both `noise_variance` and `likelihood`."
        if likelihood is None:
            if noise_variance is None:
                noise_variance = 1.0
            likelihood = Gaussian(noise_variance)
        X_data, Y_data = data_input_to_tensor(data)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable: InducingPoints = inducingpoint_wrapper(inducing_variable)

    @check_shapes(
        "return: []",
    )
    def upper_bound(self) -> tf.Tensor:
        """
        Upper bound for the sparse GP regression marginal likelihood.  Note that
        the same inducing points are used for calculating the upper bound, as are
        used for computing the likelihood approximation. This may not lead to the
        best upper bound. The upper bound can be tightened by optimising Z, just
        like the lower bound. This is especially important in FITC, as FITC is
        known to produce poor inducing point locations. An optimisable upper bound
        can be found in https://github.com/markvdw/gp_upper.

        The key reference is :cite:t:`titsias_2014`.

        The key quantity, the trace term, can be computed via

        >>> _, v = conditionals.conditional(X, model.inducing_variable.Z, model.kernel,
        ...                                 np.zeros((model.inducing_variable.num_inducing, 1)))

        which computes each individual element of the trace term.
        """
        X_data, Y_data = self.data

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)  # [N]
        sigma = tf.sqrt(sigma_sq)  # [N]

        Kdiag = self.kernel(X_data, full_cov=False)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)

        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())

        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True)

        A_sigma = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        AAT_sigma = tf.linalg.matmul(A_sigma, A_sigma, transpose_b=True)
        B = I + AAT_sigma
        LB = tf.linalg.cholesky(B)

        # Using the Trace bound, from Titsias' presentation
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A))

        # Alternative bound on max eigenval:
        cn_var = sigma_sq + c
        cn_std = tf.sqrt(cn_var)

        const = -0.5 * tf.reduce_sum(tf.math.log(2 * np.pi * sigma_sq))
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        A_cn = tf.linalg.triangular_solve(L, kuf / cn_std, lower=True)
        AAT_cn = tf.linalg.matmul(A_cn, A_cn, transpose_b=True)

        err = Y_data - self.mean_function(X_data)
        LC = tf.linalg.cholesky(I + AAT_cn)
        v = tf.linalg.triangular_solve(
            LC, tf.linalg.matmul(A_cn, err / cn_std[:, None]), lower=True
        )
        quad = -0.5 * tf.reduce_sum(tf.square(err / cn_std[:, None])) + 0.5 * tf.reduce_sum(
            tf.square(v)
        )

        return const + logdet + quad


class SGPR_deprecated(SGPRBase_deprecated):
    """
    Sparse Variational GP regression.

    The key reference is :cite:t:`titsias2009variational`.
    """

    class CommonTensors(NamedTuple):
        sigma_sq: tf.Tensor
        sigma: tf.Tensor
        A: tf.Tensor
        B: tf.Tensor
        LB: tf.Tensor
        AAT: tf.Tensor
        L: tf.Tensor

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return.sigma_sq: [N]",
        "return.sigma: [N]",
        "return.A: [M, N]",
        "return.B: [M, M]",
        "return.LB: [M, M]",
        "return.AAT: [M, M]",
    )
    def _common_calculation(self) -> "SGPR.CommonTensors":
        """
        Matrices used in log-det calculation

        :return:
            * :math:`σ²`,
            * :math:`σ`,
            * :math:`A = L⁻¹K_{uf}/σ`, where :math:`LLᵀ = Kᵤᵤ`,
            * :math:`B = AAT+I`,
            * :math:`LB` where :math`LBLBᵀ = B`,
            * :math:`AAT = AAᵀ`,
        """
        x, _ = self.data  # [N]
        iv = self.inducing_variable  # [M]

        sigma_sq = tf.squeeze(self.likelihood.variance_at(x), axis=-1)  # [N]
        sigma = tf.sqrt(sigma_sq)  # [N]

        kuf = Kuf(iv, self.kernel, x)  # [M, N]
        kuu = Kuu(iv, self.kernel, jitter=default_jitter())  # [M, M]
        L = tf.linalg.cholesky(kuu)  # [M, M]

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = add_noise_cov(AAT, tf.cast(1.0, AAT.dtype))
        LB = tf.linalg.cholesky(B)

        return self.CommonTensors(sigma_sq, sigma, A, B, LB, AAT, L)

    @check_shapes(
        "return: []",
    )
    def logdet_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        r"""
        Bound from Jensen's Inequality:

        .. math::
            \log |K + σ²I| <= \log |Q + σ²I| + N * \log (1 + \textrm{tr}(K - Q)/(σ²N))

        :param common: A named tuple containing matrices that will be used
        :return: log_det, lower bound on :math:`-.5 * \textrm{output_dim} * \log |K + σ²I|`
        """
        sigma_sq = common.sigma_sq
        LB = common.LB
        AAT = common.AAT

        x, y = self.data
        outdim = to_default_float(tf.shape(y)[1])
        kdiag = self.kernel(x, full_cov=False)

        # tr(K) / σ²
        trace_k = tf.reduce_sum(kdiag / sigma_sq)
        # tr(Q) / σ²
        trace_q = tf.reduce_sum(tf.linalg.diag_part(AAT))
        # tr(K - Q) / σ²
        trace = trace_k - trace_q

        # 0.5 * log(det(B))
        half_logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        # sum log(σ²)
        log_sigma_sq = tf.reduce_sum(tf.math.log(sigma_sq))

        logdet_k = -outdim * (half_logdet_b + 0.5 * log_sigma_sq + 0.5 * trace)
        return logdet_k

    @check_shapes(
        "return: []",
    )
    def quad_term(self, common: "SGPR.CommonTensors") -> tf.Tensor:
        """
        :param common: A named tuple containing matrices that will be used
        :return: Lower bound on -.5 yᵀ(K + σ²I)⁻¹y
        """
        sigma = common.sigma
        A = common.A
        LB = common.LB

        x, y = self.data
        err = (y - self.mean_function(x)) / sigma[..., None]

        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)

        # σ⁻² yᵀy
        err_inner_prod = tf.reduce_sum(tf.square(err))
        c_inner_prod = tf.reduce_sum(tf.square(c))

        quad = -0.5 * (err_inner_prod - c_inner_prod)
        return quad

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        common = self._common_calculation()
        output_shape = tf.shape(self.data[-1])
        num_data = to_default_float(output_shape[0])
        output_dim = to_default_float(output_shape[1])
        const = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        logdet = self.logdet_term(common)
        quad = self.quad_term(common)
        return const + logdet + quad

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        # could copy into posterior into a fused version

        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        sigma = tf.sqrt(sigma_sq)

        L = tf.linalg.cholesky(kuu)  # cache alpha, qinv
        A = tf.linalg.triangular_solve(L, kuf / sigma, lower=True)
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(
            num_inducing, dtype=default_float()
        )  # cache qinv
        LB = tf.linalg.cholesky(B)  # cache alpha
        Aerr = tf.linalg.matmul(A, err / sigma[..., None])
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True)
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean + self.mean_function(Xnew), var

    @check_shapes(
        "return[0]: [M, P]",
        "return[1]: [M, M]",
    )
    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs.

        SVGP with this q(u) should predict identically to SGPR.

        :return: mu, cov
        """
        X_data, Y_data = self.data

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        var = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)
        std = tf.sqrt(var)
        scaled_kuf = kuf / std
        sig = kuu + tf.matmul(scaled_kuf, scaled_kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)

        sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
        err = Y_data - self.mean_function(X_data)
        scaled_err = err / std[..., None]
        mu = tf.linalg.matmul(
            sig_sqrt_kuu,
            tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(scaled_kuf, scaled_err)),
            transpose_a=True,
        )

        return mu, cov


class GPRFITC(SGPRBase_deprecated):
    """
    This implements GP regression with the FITC approximation.

    The key reference is :cite:t:`Snelson06sparsegaussian`.

    Implementation loosely based on code from GPML matlab library although
    obviously gradients are automatic in GPflow.
    """

    @check_shapes(
        "return[0]: [N, R]",
        "return[1]: [N]",
        "return[2]: [M, M]",
        "return[3]: [M, M]",
        "return[4]: [M, R]",
        "return[5]: [N, R]",
        "return[6]: [M, R]",
    )
    def common_terms(
        self,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)  # size [N, R]
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        sigma_sq = tf.squeeze(self.likelihood.variance_at(X_data), axis=-1)

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + sigma_sq

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [M, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [M, R]

        return err, nu, Luu, L, alpha, beta, gamma

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.fitc_log_marginal_likelihood()

    @check_shapes(
        "return: []",
    )
    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu kuu^{-1} kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #             = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #             = \diag( \nu^{-1} )
        #                 - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1}
        #                     V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* (
        #    \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha )
        # )

        err, nu, _Luu, L, _alpha, _beta, gamma = self.common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(
            tf.square(err) / tf.expand_dims(nu, 1)
        ) + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                     = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                     = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                     = \log [
        #                        \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T )
        #                     ]

        constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L))
        )
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent_gps

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        _, _, Luu, L, _, _, gamma = self.common_terms()
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N]

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                - tf.linalg.matmul(w, w, transpose_a=True)
                + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(w), 0)
                + tf.reduce_sum(tf.square(intermediateA), 0)
            )  # [N, P]
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var


class SGPR_with_posterior(SGPR_deprecated):
    """
    Sparse Variational GP regression.
    The key reference is :cite:t:`titsias2009variational`.

    This is an implementation of SGPR that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.SGPRPosterior:
        """
        Create the Posterior object which contains precomputed matrices for
        faster prediction.

        precompute_cache has three settings:

        - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
          quantities and stores them as tensors (which allows differentiating
          through the prediction). This is the default.
        - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
          quantities and stores them as variables, which allows for updating
          their values without changing the compute graph (relevant for AOT
          compilation).
        - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
          immediate cache computation. This is useful for avoiding extraneous
          computations when you only want to call the posterior's
          `fused_predict_f` method.
        """

        return posteriors.SGPRPosterior(
            kernel=self.kernel,
            data=self.data,
            inducing_variable=self.inducing_variable,
            likelihood=self.likelihood,
            num_latent_gps=self.num_latent_gps,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, GPR's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class SGPR(SGPR_with_posterior):
    # subclassed to ensure __class__ == "SGPR"
    pass
