# Copyright 2016 the GPflow authors.
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

from .. import covariances, kernels, likelihoods
from ..base import Parameter
from ..config import default_float, default_jitter
from ..expectations import expectation
from ..kernels import Kernel
from ..mean_functions import MeanFunction, Zero
from ..probability_distributions import DiagonalGaussian
from ..utilities import positive
from ..utilities.ops import pca_reduce
from .gpr import GPR
from .model import GPModel
from .util import inducingpoint_wrapper


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """
    def __init__(self,
                 data: tf.Tensor,
                 latent_dim: int,
                 x_data_mean: Optional[tf.Tensor] = None,
                 kernel: Optional[Kernel] = None,
                 mean_function: Optional[MeanFunction] = None):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.

        :param data: y data matrix, size N (number of points) x D (dimensions)
        :param latent_dim: the number of latent dimensions (Q)
        :param x_data_mean: latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param mean_function: mean function, by default None.
        """
        if x_data_mean is None:
            x_data_mean = pca_reduce(data, latent_dim)

        num_latent = x_data_mean.shape[1]
        if num_latent != latent_dim:
            msg = 'Passed in number of latent {0} does not match initial X {1}.'
            raise ValueError(msg.format(latent_dim, num_latent))

        if mean_function is None:
            mean_function = Zero()

        if kernel is None:
            kernel = kernels.SquaredExponential(lengthscale=tf.ones((latent_dim, )))

        if data.shape[1] < num_latent:
            raise ValueError('More latent dimensions than observed.')

        gpr_data = (Parameter(x_data_mean), data)
        super().__init__(gpr_data, kernel, mean_function=mean_function)


class BayesianGPLVM(GPModel):
    def __init__(self,
                 data: tf.Tensor,
                 x_data_mean: tf.Tensor,
                 x_data_var: tf.Tensor,
                 kernel: Kernel,
                 num_inducing_variables: Optional[int] = None,
                 inducing_variable=None,
                 x_prior_mean=None,
                 x_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x D (dimensions)
        :param x_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param x_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
            random permutation of x_data_mean.
        :param x_prior_mean: prior mean used in KL term of bound. By default 0. Same size as x_data_mean.
        :param x_prior_var: pripor variance used in KL term of bound. By default 1.
        """
        super().__init__(kernel, likelihoods.Gaussian())
        self.data = data
        assert x_data_var.ndim == 2

        self.x_data_mean = Parameter(x_data_mean)
        self.x_data_var = Parameter(x_data_var, transform=positive())

        self.num_data, self.num_latent = x_data_mean.shape
        self.output_dim = data.shape[-1]

        assert np.all(x_data_mean.shape == x_data_var.shape)
        assert x_data_mean.shape[0] == data.shape[0], 'X mean and Y must be same size.'
        assert x_data_var.shape[0] == data.shape[0], 'X var and Y must be same size.'

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError("BayesianGPLVM needs exactly one of `inducing_variable` and `num_inducing_variables`")

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            inducing_variable = np.random.permutation(x_data_mean.copy())[:num_inducing_variables]

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert x_data_mean.shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if x_prior_mean is None:
            x_prior_mean = tf.zeros((self.num_data, self.num_latent), dtype=default_float())
        if x_prior_var is None:
            x_prior_var = tf.ones((self.num_data, self.num_latent))

        self.x_prior_mean = tf.convert_to_tensor(np.atleast_1d(x_prior_mean), dtype=default_float())
        self.x_prior_var = tf.convert_to_tensor(np.atleast_1d(x_prior_var), dtype=default_float())

        assert self.x_prior_mean.shape[0] == self.num_data
        assert self.x_prior_mean.shape[1] == self.num_latent
        assert self.x_prior_var.shape[0] == self.num_data
        assert self.x_prior_var.shape[1] == self.num_latent

    def log_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        pX = DiagonalGaussian(self.x_data_mean, self.x_data_var)

        y_data = self.data
        num_inducing = len(self.inducing_variable)
        psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(expectation(pX, (self.kernel, self.inducing_variable),
                                         (self.kernel, self.inducing_variable)),
                             axis=0)
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, y_data), lower=True) / sigma

        # KL[q(x) || p(x)]
        dx_data_var = self.x_data_var if self.x_data_var.shape.ndims == 2 else tf.linalg.diag_part(self.x_data_var)
        NQ = tf.cast(tf.size(self.x_data_mean), default_float())
        D = tf.cast(tf.shape(y_data)[1], default_float())
        KL = -0.5 * tf.reduce_sum(tf.math.log(dx_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.x_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum((tf.square(self.x_data_mean - self.x_prior_mean) + dx_data_var) / self.x_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(y_data), default_float())
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        return bound

    def predict_f(self, predict_at: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False):
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.

        Note: This model does not allow full output covariances.

        :param predict_at: Point to predict at.
        """
        assert full_output_cov == False
        pX = DiagonalGaussian(self.x_data_mean, self.x_data_var)

        y_data = self.data
        num_inducing = len(self.inducing_variable)
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(expectation(pX, (self.kernel, self.inducing_variable),
                                         (self.kernel, self.inducing_variable)),
                             axis=0)
        jitter = default_jitter()
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, predict_at)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, y_data), lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kernel(predict_at) + tf.linalg.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kernel(predict_at, full=False) + tf.reduce_sum(tf.square(tmp2), 0) - tf.reduce_sum(
                tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(predict_at), var
