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

from typing import Optional

import numpy as np
import tensorflow as tf

import gpflow

from .. import posteriors
from ..base import InputData, MeanAndVariance, Parameter, RegressionData
from ..conditionals import conditional
from ..config import default_float, default_jitter
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..kernels import Kernel
from ..kullback_leiblers import gauss_kl
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction
from ..utilities import assert_params_false, is_variable, triangular, triangular_size
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor


class VGP_deprecated(GPModel, InternalDataTrainingLossMixin):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.

    This implementation is equivalent to SVGP with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\mathbf f) = N(\mathbf f \,|\, \boldsymbol \mu, \boldsymbol \Sigma)

    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
    ):
        """
        data = (X, Y) contains the input points [N, D] and the observations [N, P]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.data = data_input_to_tensor(data)
        X_data, _Y_data = self.data

        static_num_data = X_data.shape[0]
        if static_num_data is None:
            q_sqrt_unconstrained_shape = (self.num_latent_gps, None)
        else:
            q_sqrt_unconstrained_shape = (self.num_latent_gps, triangular_size(static_num_data))
        self.num_data = Parameter(tf.shape(X_data)[0], shape=[], dtype=tf.int32, trainable=False)

        # Many functions below don't like `Parameter`s:
        dynamic_num_data = tf.convert_to_tensor(self.num_data)

        self.q_mu = Parameter(
            tf.zeros((dynamic_num_data, self.num_latent_gps)),
            shape=(static_num_data, num_latent_gps),
        )
        q_sqrt = tf.eye(dynamic_num_data, batch_shape=[self.num_latent_gps])
        self.q_sqrt = Parameter(
            q_sqrt,
            transform=triangular(),
            unconstrained_shape=q_sqrt_unconstrained_shape,
            constrained_shape=(num_latent_gps, static_num_data, static_num_data),
        )

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        r"""
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\mathbf f) = N(\mathbf f \,|\, \boldsymbol \mu, \boldsymbol \Sigma)

        """
        X_data, Y_data = self.data
        num_data = tf.convert_to_tensor(self.num_data)

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kernel(X_data) + tf.eye(num_data, dtype=default_float()) * default_jitter()
        L = tf.linalg.cholesky(K)
        fmean = tf.linalg.matmul(L, self.q_mu) + self.mean_function(X_data)  # [NN, ND] -> ND
        q_sqrt_dnn = tf.linalg.band_part(self.q_sqrt, -1, 0)  # [D, N, N]
        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent_gps, 1, 1]))
        LTA = tf.linalg.matmul(L_tiled, q_sqrt_dnn)  # [D, N, N]
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(X_data, fmean, fvar, Y_data)

        return tf.reduce_sum(var_exp) - KL

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, _Y_data = self.data
        mu, var = conditional(
            Xnew,
            X_data,
            self.kernel,
            self.q_mu,
            q_sqrt=self.q_sqrt,
            full_cov=full_cov,
            white=True,
        )
        return mu + self.mean_function(Xnew), var


class VGP_with_posterior(VGP_deprecated):
    """
    This is an implementation of VGP that provides a posterior() method that
    enables caching for faster subsequent predictions.
    """

    def posterior(
        self,
        precompute_cache: posteriors.PrecomputeCacheType = posteriors.PrecomputeCacheType.TENSOR,
    ) -> posteriors.VGPPosterior:
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
        X_data, _Y_data = self.data
        return posteriors.VGPPosterior(
            self.kernel,
            X_data,
            self.q_mu,
            self.q_sqrt,
            mean_function=self.mean_function,
            precompute_cache=precompute_cache,
        )

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        For backwards compatibility, VGP's predict_f uses the fused (no-cache)
        computation, which is more efficient during training.

        For faster (cached) prediction, predict directly from the posterior object, i.e.,:
            model.posterior().predict_f(Xnew, ...)
        """
        return self.posterior(posteriors.PrecomputeCacheType.NOCACHE).fused_predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


class VGP(VGP_with_posterior):
    # subclassed to ensure __class__ == "VGP"
    pass


@check_shapes(
    "new_data[0]: [N, D]",
    "new_data[1]: [N, P]",
)
def update_vgp_data(vgp: VGP_deprecated, new_data: RegressionData) -> None:
    """
    Set the data on the given VGP model, and update its variational parameters.

    As opposed to many of the other models the VGP has internal parameters whose shape depends on
    the shape of the data. This functions updates the internal data of the given vgp, and updates
    the variational parameters to fit.

    This function requires that the input `vgp` were create with :class:`tf.Variable`s for
    `data`.
    """
    old_X_data, old_Y_data = vgp.data
    assert is_variable(old_X_data) and is_variable(
        old_Y_data
    ), "update_vgp_data requires the model to have been created with variable data."

    new_X_data, new_Y_data = new_data
    new_num_data = tf.shape(new_X_data)[0]
    f_mu, f_cov = vgp.predict_f(new_X_data, full_cov=True)  # [N, L], [L, N, N]

    # This model is hard-coded to use the whitened representation, i.e.  q_mu and q_sqrt
    # parametrize q(v), and u = f(X) = L v, where L = cholesky(K(X, X)) Hence we need to
    # back-transform from f_mu and f_cov to obtain the updated new_q_mu and new_q_sqrt:
    Knn = vgp.kernel(new_X_data, full_cov=True)  # [N, N]
    jitter_mat = default_jitter() * tf.eye(new_num_data, dtype=Knn.dtype)
    Lnn = tf.linalg.cholesky(Knn + jitter_mat)  # [N, N]
    new_q_mu = tf.linalg.triangular_solve(Lnn, f_mu)  # [N, L]
    tmp = tf.linalg.triangular_solve(Lnn[None], f_cov)  # [L, N, N], L⁻¹ f_cov
    S_v = tf.linalg.triangular_solve(Lnn[None], tf.linalg.matrix_transpose(tmp))  # [L, N, N]
    new_q_sqrt = tf.linalg.cholesky(S_v + jitter_mat)  # [L, N, N]

    old_X_data.assign(new_X_data)
    old_Y_data.assign(new_Y_data)
    vgp.num_data.assign(new_num_data)
    vgp.q_mu.assign(new_q_mu)
    vgp.q_sqrt.assign(new_q_sqrt)


class VGPOpperArchambeau(GPModel, InternalDataTrainingLossMixin):
    r"""
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The key reference is :cite:t:`Opper:2009`.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior. It turns out that the optimal
    posterior precision shares off-diagonal elements with the prior, so
    only the diagonal elements of the precision need be adjusted.
    The posterior approximation is

    .. math::

       q(\mathbf f) = N(\mathbf f \,|\, \mathbf K \boldsymbol \alpha,
                         [\mathbf K^{-1} + \textrm{diag}(\boldsymbol \lambda))^2]^{-1})

    This approach has only 2ND parameters, rather than the N + N^2 of vgp,
    but the optimization is non-convex and in practice may cause difficulty.

    """

    @check_shapes(
        "data[0]: [N, D]",
        "data[1]: [N, P]",
    )
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        likelihood: Likelihood,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
    ):
        """
        data = (X, Y) contains the input points [N, D] and the observations [N, P]
        kernel, likelihood, mean_function are appropriate GPflow objects
        """
        if num_latent_gps is None:
            num_latent_gps = self.calc_num_latent_gps_from_data(data, kernel, likelihood)
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)

        self.data = data_input_to_tensor(data)
        X_data, _Y_data = self.data
        self.num_data = X_data.shape[0]
        self.q_alpha = Parameter(np.zeros((self.num_data, self.num_latent_gps)))
        self.q_lambda = Parameter(
            np.ones((self.num_data, self.num_latent_gps)), transform=gpflow.utilities.positive()
        )

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        r"""
        q_alpha, q_lambda are variational parameters, size [N, R]
        This method computes the variational lower bound on the likelihood,
        which is:

        .. math::

           E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

        .. math::

           q(f) = N(f |
               K \alpha + \textrm{mean},
               [K^-1 + \textrm{diag}(\textrm{square}(\lambda))]^-1) .
        """
        X_data, Y_data = self.data

        K = self.kernel(X_data)
        K_alpha = tf.linalg.matmul(K, self.q_alpha)
        f_mean = K_alpha + self.mean_function(X_data)

        # compute the variance for each of the outputs
        I = tf.tile(
            tf.eye(self.num_data, dtype=default_float())[None, ...], [self.num_latent_gps, 1, 1]
        )
        A = (
            I
            + tf.transpose(self.q_lambda)[:, None, ...]
            * tf.transpose(self.q_lambda)[:, :, None, ...]
            * K
        )
        L = tf.linalg.cholesky(A)
        Li = tf.linalg.triangular_solve(L, I)
        tmp = Li / tf.transpose(self.q_lambda)[:, None, ...]
        f_var = 1.0 / tf.square(self.q_lambda) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

        # some statistics about A are used in the KL
        A_logdet = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (
            A_logdet
            + trAi
            - self.num_data * self.num_latent_gps
            + tf.reduce_sum(K_alpha * self.q_alpha)
        )

        v_exp = self.likelihood.variational_expectations(X_data, f_mean, f_var, Y_data)
        return tf.reduce_sum(v_exp) - KL

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        The posterior variance of F is given by

        .. math::

           q(f) = N(f |
               K \alpha + \textrm{mean}, [K^-1 + \textrm{diag}(\lambda**2)]^-1)

        Here we project this to F*, the values of the GP at Xnew which is given
        by

        .. math::

           q(F*) = N ( F* | K_{*F} \alpha + \textrm{mean}, K_{**} - K_{*f}[K_{ff} +
                                           \textrm{diag}(\lambda**-2)]^-1 K_{f*} )

        Note: This model currently does not allow full output covariances
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        X_data, _ = self.data
        # compute kernel things
        Kx = self.kernel(X_data, Xnew)
        K = self.kernel(X_data)

        # predictive mean
        f_mean = tf.linalg.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # predictive var
        A = K + tf.linalg.diag(tf.transpose(1.0 / tf.square(self.q_lambda)))
        L = tf.linalg.cholesky(A)
        Kx_tiled = tf.tile(Kx[None, ...], [self.num_latent_gps, 1, 1])
        LiKx = tf.linalg.triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kernel(Xnew) - tf.linalg.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kernel(Xnew, full_cov=False) - tf.reduce_sum(tf.square(LiKx), axis=1)
        return f_mean, tf.transpose(f_var)
