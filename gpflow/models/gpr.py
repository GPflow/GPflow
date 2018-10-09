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

import tensorflow as tf

from .. import likelihoods
from .. import settings

from ..conditionals import base_conditional
from ..logdensities import multivariate_normal

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
    def __init__(self, X, Y, kernel, mean_function=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()

        def convert_to_tensor(x):
            if isinstance(x, (tf.Tensor, tfe.Variable)):
                return x
            return tf.convert_to_tensor(x)

        self.X = convert_to_tensor(X)
        self.Y = convert_to_tensor(Y)
        GPModel.__init__(self, kernel, likelihood, mean_function)

    def log_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kernel(self.X) + tf.eye(self.X.shape[0], dtype=X.dtype) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y
        return tf.reduce_sum(logpdf)

    def predict_f(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kernel(self.X, Xnew)
        Kmm_sigma = self.kernel(self.X) + tf.eye(self.X.shape[0], dtype=X.dtype) * self.likelihood.variance
        Knn = self.kernel(Xnew, diag=(not full_cov))
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # [N, P], [N, P] or [P, N, N]
        return f_mean + self.mean_function(Xnew), f_var

