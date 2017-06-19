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

from __future__ import absolute_import
import tensorflow as tf
import numpy as np
from .param import Param, DataHolder
from .model import GPModel
from .mean_functions import Zero
from ._settings import settings
float_type = settings.dtypes.float_type
from .conditionals import conditional
from .kullback_leiblers import  gauss_kl_white

class VGP(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior.
    
    This implementation is equivalent to svgp with X=Z, but is more efficient.
    The whitened representation is used to aid optimization.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)
       
    """
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """

        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]

        self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
        self.q_sqrt = Param(np.eye(self.num_data)[:, :, None] * 
                            np.ones((1, 1, self.num_latent)))
        
    def _compile(self, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_sqrt = Param(np.eye(self.num_data)[:, :, None] * 
                                np.ones((1, 1, self.num_latent)))

        return super(VGP, self)._compile(optimizer=optimizer)

    def build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

        """

        # Get prior KL.
        KL = gauss_kl_white(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kern.K(self.X) + tf.eye(self.num_data, dtype=float_type) * settings.numerics.jitter_level
        L = tf.cholesky(K)

        fmean = tf.matmul(L, self.q_mu) + self.mean_function(self.X) # NN,ND->ND
        
        q_sqrt_dnn = tf.matrix_band_part(tf.transpose(self.q_sqrt, [2, 0, 1]), -1, 0)  # D x N x N

        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))
        
        LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)
        
        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        return tf.reduce_sum(var_exp) - KL

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=True)
        return mu + self.mean_function(Xnew), var
