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

from typing import Optional, Tuple

import tensorflow as tf

import gpflow

from ..kernels import Kernel
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from .model import GPModel, InputData, MeanAndVariance, RegressionData
from .training_mixins import InternalDataTrainingLossMixin
from .util import data_input_to_tensor
from ..config import default_jitter

class GPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is given by

    .. math::
       \log p(Y \,|\, \mathbf f) =
            \mathcal N(Y \,|\, 0, \sigma_n^2 \mathbf{I})
            
    To train the model, we maximise the log _marginal_ likelihood
    w.r.t. the likelihood variance and kernel hyperparameters theta.
    The marginal likelihood is found by integrating the likelihood
    over the prior, and has the form
    
    .. math::
       \log p(Y \,|\, \sigma_n, \theta) =
            \mathcal N(Y \,|\, 0, \mathbf{K} + \sigma_n^2 \mathbf{I})
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.

        .. math::
            \log p(Y | \theta).

        """
        X, Y = self.data
        K = self.kernel(X)
        num_data = tf.shape(X)[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(X)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)

    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)

        kmm = self.kernel(X_data)
        knn = self.kernel(Xnew, full_cov=full_cov)
        kmn = self.kernel(X_data, Xnew)

        num_data = X_data.shape[0]
        s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(
            kmn, kmm + s, knn, err, full_cov=full_cov, white=False
        )  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(Xnew)
        return f_mean, f_var

class MultivariateGPR(GPModel, InternalDataTrainingLossMixin):
    r"""
    Multivariate extension of the vanilla Gaussian Process Regression.
    """
    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        mean_function: Optional[MeanFunction] = None,
        noise_variance: float = 1.0,
    ):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, Y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=Y_data.shape[-1])
        self.data = data_input_to_tensor(data)

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.log_marginal_likelihood()

    def log_marginal_likelihood(self) -> tf.Tensor:
        r"""
        Computes the log marginal likelihood.
        .. math::
            \log p(Y | \theta).
        """
        X, Y = self.data
        m = self.mean_function(X) # [N, 1]
        Kmm = self.kernel(X, full_cov = True, full_output_cov = True) # [M, P, M, P]

        # reshape to be compatible with multi output
        M, P = tf.shape(Y)[0], tf.shape(Y)[1]     
        Y = tf.reshape(Y, shape = (M * P, 1)) # [M*P, 1]
        m = tf.tile(m, (P, tf.constant(1))) # [M*P, 1]
        Kmm = tf.reshape(Kmm, shape = (M * P, M * P)) # [M*P, M*P]
        # add jitter to the diagional
        Kmm += default_jitter() * tf.eye(M * P, dtype=Kmm.dtype)
        Kmm += tf.linalg.diag(tf.fill([M * P], self.likelihood.variance)) # [M*P, M*P]
        L = tf.linalg.cholesky(Kmm)  # [M*P, M*P]

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(Y, m, L)
        return tf.reduce_sum(log_prob)
    
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        r"""
        This method computes predictions at X \in R^{N \x D} input points
        .. math::
            p(F* | Y)
        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        X_data, Y_data = self.data
        err = Y_data - self.mean_function(X_data)
        
        Kmm = self.kernel(X_data, full_cov = True, full_output_cov = True) # [M, P, M, P]
        Knn = self.kernel(Xnew, full_cov=full_cov, full_output_cov = full_output_cov) # [N, P, N, P] or [N, N, P] or [P, N, N] or [N, P] 
        Kmn = self.kernel(X_data, Xnew, full_cov = True, full_output_cov = True) # [M, P, N, P]

        M, P, N, _ = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Kmn = tf.reshape(Kmn, (M * P, N, P)) # [M*P, N, P]
        err = tf.reshape(err, shape = (M * P, 1)) # [M*P, 1]
        Kmm = tf.reshape(Kmm, (M * P, M * P)) # [M*P, M*P]
        Kmm += tf.linalg.diag(tf.fill([M * P], self.likelihood.variance)) # [M*P, M*P]
        # add jitter to the diagional
        Kmm += default_jitter() * tf.eye(M * P, dtype=Kmm.dtype)
        
        M, N, P = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
        Lm = tf.linalg.cholesky(Kmm) # [M*P, M*P]
        Kmn = tf.reshape(Kmn, (M, N * P))  # [M*P, N*P]
        A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [M*P, N*P]
        Ar = tf.reshape(A, (M, N, P)) # [M*P, N, P]

        # compute the covariance due to the conditioning
        if full_cov and full_output_cov:
            fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # [N, P, N, P]
        elif full_cov and not full_output_cov:
            At = tf.transpose(Ar)  # [P, N, M*P]
            fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [P, N, N]
        elif not full_cov and full_output_cov:
            At = tf.transpose(Ar, [1, 0, 2])  # [N, M*P, P]
            fvar = Knn - tf.linalg.matmul(At, At, transpose_a=True)  # [N, P, P]
        elif not full_cov and not full_output_cov:
            fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0]), (N, P)) # [N, P]
        
        # another backsubstitution in the unwhitened case
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)  # [M*P, N*P]
        fmean_zero = tf.linalg.matmul(err, A, transpose_a=True)  # [1, N*P]
        fmean_zero = tf.reshape(fmean_zero, (N, P))  # [N, P]
        return fmean_zero + self.mean_function(Xnew), fvar
