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
from ..params import DataHolder
from ..decors import params_as_tensors
from ..decors import name_scope
from ..logdensities import multivariate_normal

from .model import GPModel


class GPR(GPModel):
    r"""
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood of this model is sometimes referred to as the
    'marginal log likelihood', and is given by

    .. math::

       \log p(\mathbf y | \mathbf f) = \mathcal N(\mathbf y | 0, \mathbf K + \sigma_n \mathbf I)
    """
    def __init__(self, X, Y, kern, mean_function=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        r"""
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)
        logpdf = multivariate_normal(self.Y, m, L)  # (R,) log-likelihoods for each independent dimension of Y

        return tf.reduce_sum(logpdf)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, the points at which we want to predict.

        This method computes

            p(F* | Y)

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        y = self.Y - self.mean_function(self.X)
        Kmn = self.kern.K(self.X, Xnew)
        Kmm_sigma = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.float_type) * self.likelihood.variance
        Knn = self.kern.K(Xnew) if full_cov else self.kern.Kdiag(Xnew)
        f_mean, f_var = base_conditional(Kmn, Kmm_sigma, Knn, y, full_cov=full_cov, white=False)  # N x P, N x P or P x N x N
        return f_mean + self.mean_function(Xnew), f_var

