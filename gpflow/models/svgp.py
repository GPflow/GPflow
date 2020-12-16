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

from typing import Tuple

import numpy as np
import tensorflow as tf

from .. import covariances, kernels, kullback_leiblers
from ..base import Module, Parameter
from ..conditionals import conditional
from ..conditionals.util import expand_independent_outputs, mix_latent_gp
from ..config import default_float, default_jitter
from ..models.model import GPModel, InputData, MeanAndVariance, RegressionData
from ..models.training_mixins import ExternalDataTrainingLossMixin
from ..models.util import inducingpoint_wrapper
from ..utilities import positive, triangular
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import ExternalDataTrainingLossMixin
from .util import inducingpoint_wrapper


class OldSVGP(GPModel, ExternalDataTrainingLossMixin):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(
        self,
        kernel,
        likelihood,
        inducing_variable,
        *,
        mean_function=None,
        num_latent_gps: int = 1,
        q_diag: bool = False,
        q_mu=None,
        q_sqrt=None,
        whiten: bool = True,
        num_data=None,
    ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent_gps is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent_gps)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        num_inducing = self.inducing_variable.num_inducing
        self._init_variational_parameters(num_inducing, q_mu, q_sqrt, q_diag)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """
        q_mu = np.zeros((num_inducing, self.num_latent_gps)) if q_mu is None else q_mu
        self.q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent_gps), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [
                    np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent_gps)
                ]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent_gps = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent_gps = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self) -> tf.Tensor:
        return kullback_leiblers.prior_kl(
            self.inducing_variable, self.kernel, self.q_mu, self.q_sqrt, whiten=self.whiten
        )

    def maximum_log_likelihood_objective(self, data: RegressionData) -> tf.Tensor:
        return self.elbo(data)

    def elbo(self, data: RegressionData) -> tf.Tensor:
        """
        This gives a variational bound (the evidence lower bound or ELBO) on
        the log marginal likelihood of the model.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        mu, var = conditional(
            Xnew,
            self.inducing_variable,
            self.kernel,
            q_mu,
            q_sqrt=q_sqrt,
            full_cov=full_cov,
            white=self.whiten,
            full_output_cov=full_output_cov,
        )
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var


class DiagNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = Parameter(q_mu)  # [M, L]
        self.q_sqrt = Parameter(q_sqrt)  # [M, L]


class MvnNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = Parameter(q_mu)  # [M, L]
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L, M, M]


def eye_like(A):
    return tf.eye(tf.shape(A)[-1], dtype=A.dtype)


class Posterior(Module):
    def __init__(self, kernel, iv, q_dist, whiten=True, mean_function=None):
        self.iv = iv
        self.kernel = kernel
        self.q_dist = q_dist
        self.mean_function = mean_function
        self.whiten = whiten

        self._precompute()  # populates self.alpha and self.Qinv

    def freeze(self):
        """
        Note- this simply cuts the computational graph
        """
        self.alpha = Parameter(self.alpha.numpy(), trainable=False)
        self.Qinv = Parameter(self.Qinv.numpy(), trainable=False)

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # Qinv: [L, M, M]
        # alpha: [M, L]

        Kuf = covariances.Kuf(self.iv, self.kernel, Xnew)  # [(R), M, N]
        mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.einsum("...nr->...rn", tf.squeeze(mean, axis=-1))

        if isinstance(self.kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):

            Knn = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        elif isinstance(self.kernel, kernels.MultioutputKernel):
            Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
        else:
            Knn = self.kernel(Xnew, full_cov=full_cov)

        if full_cov:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, tf.matmul(self.Qinv, Kuf), transpose_a=True)
            cov = Knn - Kfu_Qinv_Kuf
        else:
            # [AT B]_ij = AT_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(self.Qinv, Kuf), axis=-2)
            cov = Knn - Kfu_Qinv_Kuf
            cov = tf.linalg.adjoint(cov)

        if isinstance(self.kernel, kernels.LinearCoregionalization):
            cov = expand_independent_outputs(cov, full_cov, full_output_cov=False)
            mean, cov = mix_latent_gp(self.kernel.W, mean, cov, full_cov, full_output_cov)
        else:
            cov = expand_independent_outputs(cov, full_cov, full_output_cov)

        return mean + self.mean_function(Xnew), cov

    def _precompute(self):
        Kuu = covariances.Kuu(self.iv, self.kernel, jitter=default_jitter())  # [(R), M, M]
        L = tf.linalg.cholesky(Kuu)

        q_mu = self.q_dist.q_mu
        if Kuu.shape.ndims == 3:
            q_mu = tf.einsum("...mr->...rm", self.q_dist.q_mu)[..., None]  # [..., R, M, 1]

        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = tf.linalg.cholesky_solve(L, q_mu)
        else:
            # alpha = L⁻T q_mu
            alpha = tf.linalg.triangular_solve(L, q_mu, adjoint=True)
        # predictive mean = Kfu alpha
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtT
        if not self.whiten:
            # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
            #      = Kuu⁻¹ - L⁻T L⁻¹ S L⁻T L⁻¹
            #      = L⁻T (I - L⁻¹ S L⁻T) L⁻¹
            #      = L⁻T B L⁻¹
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
            # Qinv = Kuu⁻¹ - L⁻T S L⁻¹
            # Linv = (L⁻¹ I) = solve(L, I)
            # Kinv = Linv.T @ Linv
        I = eye_like(Linv_cov_u_LinvT)
        B = I - Linv_cov_u_LinvT
        LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        B_Linv = tf.linalg.adjoint(LinvT_B)
        Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)
        self.alpha = alpha
        self.Qinv = Qinv


class NewSVGP(OldSVGP):
    """
    Adds posterior() method and uses different math ordering for predict_f
    """

    def posterior(self, freeze=False):
        """
        If freeze=True, cuts the computational graph after precomputing alpha and Qinv
        this works around some issues in the tensorflow graph optimisation and gives much
        faster prediction when wrapped inside tf.function()
        """
        if self.q_diag:
            q_dist = DiagNormal(self.q_mu, self.q_sqrt)
        else:
            q_dist = MvnNormal(self.q_mu, self.q_sqrt)
        posterior = Posterior(
            self.kernel,
            self.inducing_variable,
            q_dist,
            whiten=self.whiten,
            mean_function=self.mean_function,
        )
        if freeze:
            posterior.freeze()
        return posterior

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        For backwards compatibility.
        For faster (cached) prediction, get a posterior object first:
            posterior = model.posterior()
        then call
            posterior.predict_f(Xnew, ...)
        """
        return self.posterior(freeze=False).predict_f(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )


SVGP = NewSVGP
