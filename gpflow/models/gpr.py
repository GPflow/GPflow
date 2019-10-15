# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
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

from .model import GPModel, GPPosterior, Data
from ..kernels import Kernel
from ..likelihoods import Gaussian
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction


class GPR(GPModel):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this models is sometimes referred to as the 'marginal log likelihood',
    and is given by

    .. math::
       \log p(\mathbf y \,|\, \mathbf f) =
            \mathcal N\left(\mathbf y\,|\, 0, \mathbf K + \sigma_n \mathbf I\right)
    """

    def __init__(self,
                 kernel: Kernel,
                 mean_function: Optional[MeanFunction] = None, 
                 noise_variance: float = 1.0
                 ) -> None:
        likelihood = Gaussian(variance=noise_variance)
        super().__init__(kernel, likelihood, mean_function)

    def log_marginal_likelihood(self, data: Data):
        r"""
        Computes the log likelihood.

        .. math::
            \log p(Y | \theta).

        """
        x, y = data
        K = self.kernel(x)
        num_data = x.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(x)

        # [R,] log-likelihoods for each independent dimension of Y
        log_prob = multivariate_normal(y, m, L)
        return tf.reduce_sum(log_prob)

    def objective(self, data: Data):
        return -self.log_marginal_likelihood(data)

    def get_posterior(self, data: Data) -> GPPosterior:
        x_data, y_data = data
        K = self.kernel(x_data)
        num_data = x_data.shape[0]
        k_diag = tf.linalg.diag_part(K)
        s_diag = tf.fill([num_data], self.likelihood.variance)
        ks = tf.linalg.set_diag(K, k_diag + s_diag)
        L = tf.linalg.cholesky(ks)
        m = self.mean_function(x_data)

        # TODO: these computations are correct but not as efficient as they might be.
        # to improve efficiency, we'd need to be able to pass chol(K) in to the conditionals code.
        # TODO (edit) the variance appears to not be correct after all...?

        tmp = tf.linalg.triangular_solve(L, y_data - m, lower=True)
        tmp = tf.linalg.triangular_solve(tf.transpose(L), tmp, lower=False)
        q_mu = tf.matmul(K, tmp)

        tmp = tf.linalg.triangular_solve(L, K, lower=True)
        variance = K - tf.matmul(tmp, tmp, transpose_a=True)
        variance_sqrt = tf.linalg.cholesky(variance).numpy()

        return GPPosterior(mean_function=self.mean_function,
                           kernel=self.kernel,
                           likelihood=self.likelihood,
                           inducing_variable=data[0],
                           whiten=False,
                           mean=q_mu,
                           variance_sqrt=variance_sqrt)
