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

from typing import Optional
import tensorflow as tf

from .. import likelihoods
from ..conditionals import base_conditional
from ..kernels import Kernel
from ..logdensities import multivariate_normal
from ..mean_functions import MeanFunction
from .model import GPModel


class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """

    def __init__(self,
                 X: tf.Tensor,
                 Y: tf.Tensor,
                 kernel: Kernel,
                 mean_function: Optional[MeanFunction] = None):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        # kernel, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()

        self.X = X
        self.Y = Y
        super().__init__(kernel, likelihood, mean_function)

    def log_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kernel(self.X)
        S = tf.eye(self.X.shape[0],
                   dtype=self.X.dtype) * self.likelihood.variance
        L = tf.linalg.cholesky(K + S)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(
            self.Y, m,
            L)  # (R,) log-likelihoods for each independent dimension of Y
        return tf.reduce_sum(logpdf)

    def predict_f(self, Xnew, full_cov=False, full_output_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        X = self.X
        y = self.Y - self.mean_function(X)
        Kmn = self.kernel(X, Xnew)
        S = tf.eye(X.shape[0], dtype=X.dtype) * self.likelihood.variance
        Kmm = self.kernel(X)
        Knn = self.kernel(Xnew, full=full_cov)
        f_mean, f_var = base_conditional(
            Kmn, Kmm + S, Knn, y, full_cov=full_cov,
            white=False)  # [N, P], [N, P] or [P, N, N]
        return f_mean + self.mean_function(Xnew), f_var
