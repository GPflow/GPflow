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

import gpflow
from .model import GPModel
from ..kernels import Kernel
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction

Data = Tuple[tf.Tensor, tf.Tensor]


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

    def __init__(self, data: Data, kernel: Kernel, mean_function: Optional[MeanFunction] = None,
                 noise_variance: float = 1.0):
        likelihood = gpflow.likelihoods.Gaussian(noise_variance)
        _, y_data = data
        super().__init__(kernel, likelihood, mean_function, num_latent=y_data.shape[-1])
        self.data = data

    def log_likelihood(self):
        r"""
        Computes the log likelihood.

        .. math::
            \log p(Y | \theta).

        """
        x, y = self.data
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

    def predict_f(self, predict_at: tf.Tensor, full_cov: bool = False, full_output_cov: bool = False):
        r"""
        This method computes predictions at X \in R^{N \x D} input points

        .. math::
            p(F* | Y)

        where F* are points on the GP at new data points, Y are noisy observations at training data points.
        """
        x_data, y_data = self.data
        err = y_data - self.mean_function(x_data)

        kmm = self.kernel(x_data)
        knn = self.kernel(predict_at, full=full_cov)
        kmn = self.kernel(x_data, predict_at)

        num_data = x_data.shape[0]
        s = tf.linalg.diag(tf.fill([num_data], self.likelihood.variance))

        conditional = gpflow.conditionals.base_conditional
        f_mean_zero, f_var = conditional(kmn, kmm + s, knn, err, full_cov=full_cov,
                                         white=False)  # [N, P], [N, P] or [P, N, N]
        f_mean = f_mean_zero + self.mean_function(predict_at)
        return f_mean, f_var
