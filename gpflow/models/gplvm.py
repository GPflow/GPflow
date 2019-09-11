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

import tensorflow as tf
import numpy as np

from .. import likelihoods
from .. import kernels
from .. import inducing_variables

from ..mean_functions import Zero
from ..expectations import expectation
from ..probability_distributions import DiagonalGaussian

from ..base import Parameter
from .model import GPModel
from .gpr import GPR


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    def __init__(self, data, latent_dim, x_data_mean=None, kernel=None, mean_function=None, **kwargs):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.

        :param data: y data matrix, size N (number of points) x D (dimensions)
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions)
        :param x_data_mean: latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param mean_function: mean function, by default None.
        """
        if x_data_mean is None:
            x_data_mean = PCA_reduce(data, latent_dim)

        num_latent = x_data_mean.shape[1]
        if num_latent != latent_dim:
            msg = 'Passed in number of latent {0} does not match initial X {1}.'
            raise ValueError(msg.format(latent_dim, num_latent))

        if mean_function is None:
            mean_function = Zero()

        if kernel is None:
            kernel = kernels.SquaredExponential(latent_dim, ARD=True)

        if data.shape[1] < num_latent:
            raise ValueError('More latent dimensions than observed.')

        data = (x_data_mean, data)
        super().__init__(self, data, kernel, mean_function=mean_function, **kwargs)
        x_parameter = Parameter(x_data_mean)
        self.data = (x_parameter, data)


class BayesianGPLVM(GPModel):
    def __init__(self, data, x_data_mean, x_data_var, kernel, M, Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param data: data matrix, size N (number of points) x D (dimensions)
        :param x_data_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param x_data_var: variance of latent positions ([N, Q]), for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param M: number of inducing points
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
        random permutation of x_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as x_data_mean.
        :param X_prior_var: pripor variance used in KL term of bound. By default 1.
        """
        super().__init__(self, data, x_data_mean=x_data_mean, kernel=kernel, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        self.x_data_mean = Parameter(x_data_mean)
        # diag_transform = transforms.DiagMatrix(x_data_var.shape[1])
        # self.x_data_var() = Parameter(diag_transform.forward(transforms.positive.backward(x_data_var)) if x_data_var.ndim == 2 else x_data_var,
        #                    diag_transform)
        assert x_data_var.ndim == 2
        self.x_data_var = Parameter(x_data_var, transform=transforms.positive)

        self.num_data, self.num_latent = x_data_mean.shape
        self.output_dim = Y.shape[1]

        assert np.all(x_data_mean.shape == x_data_var.shape)
        assert x_data_mean.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        assert x_data_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            Z = np.random.permutation(x_data_mean.copy())[:M]

        self.inducing_variable = inducing_variables.InducingPoints(Z)

        assert len(self.inducing_variable) == M
        assert x_data_mean.shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = np.zeros((self.num_data, self.num_latent))
        if X_prior_var is None:
            X_prior_var = np.ones((self.num_data, self.num_latent))

        self.X_prior_mean = np.asarray(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = np.asarray(np.atleast_1d(X_prior_var), dtype=default_float())

        assert self.X_prior_mean.shape[0] == self.num_data
        assert self.X_prior_mean.shape[1] == self.num_latent
        assert self.X_prior_var.shape[0] == self.num_data
        assert self.X_prior_var.shape[1] == self.num_latent

    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        pX = DiagonalGaussian(self.x_data_mean(), self.x_data_var())

        num_inducing = len(self.inducing_variable)
        psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(expectation(pX, (self.kernel, self.inducing_variable),
                                         (self.kernel, self.inducing_variable)),
                             axis=0)
        cov_uu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
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
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        dx_data_var = self.x_data_var() if len(self.x_data_var().get_shape()) == 2 else tf.linalg.diag_part(self.x_data_var())
        NQ = tf.cast(tf.size(self.x_data_mean()), default_float())
        D = tf.cast(tf.shape(self.Y)[1], default_float())
        KL = -0.5 * tf.reduce_sum(tf.math.log(dx_data_var)) \
             + 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var)) \
             - 0.5 * NQ \
             + 0.5 * tf.reduce_sum((tf.square(self.x_data_mean() - self.X_prior_mean) + dx_data_var) / self.X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), default_float())
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        return bound

    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        :param Xnew: Point to predict at.
        """
        pX = DiagonalGaussian(self.x_data_mean(), self.x_data_var())

        num_inducing = len(self.inducing_variable)
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(expectation(pX, (self.kernel, self.inducing_variable),
                                         (self.kernel, self.inducing_variable)),
                             axis=0)
        jitter = default_jitter()
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.linalg.cholesky(Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kernel(Xnew) + tf.linalg.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kernel(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var


def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size [N, Q].
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evals, evecs = tf.linalg.eigh(np.cov(X.T))
    W = evecs[:, -Q:]
    return (X - X.mean(0)).dot(W)
