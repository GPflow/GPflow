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
from . import transforms
from .mean_functions import Zero
from .tf_wraps import eye


class VGP(GPModel):
    """
    This method approximates the Gaussian process posterior using a multivariate Gaussian.
    The key reference is:

    ::

      @article{Opper:2009,
          title = {The Variational Gaussian Approximation Revisited},
          author = {Opper, Manfred and Archambeau, Cedric},
          journal = {Neural Comput.},
          year = {2009},
          pages = {786--792},
      }

    The idea is that the posterior over the function-value vector F is
    approximated by a Gaussian, and the KL divergence is minimised between
    the approximation and the posterior. It turns out that the optimal
    posterior precision shares off-diagonal elements with the prior, so
    only the diagonal elements of the precision need be adjusted.

    The posterior approximation is

    .. math::

       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\mathbf K \\boldsymbol \\alpha, [\\mathbf K^{-1} + \\textrm{diag}(\\boldsymbol \\lambda))^2]^{-1})
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
        self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
        self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                              transforms.positive)

    def _compile(self, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the hape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                                  transforms.positive)
        return super(VGP, self)._compile(optimizer=optimizer)

    def build_likelihood(self):
        """
        q_alpha, q_lambda are variational parameters, size N x R

        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(f) = N(f | K alpha + mean, [K^-1 + diag(square(lambda))]^-1) .

        """
        K = self.kern.K(self.X)
        K_alpha = tf.matmul(K, self.q_alpha)
        f_mean = K_alpha + self.mean_function(self.X)

        # compute the variance for each of the outputs
        I = tf.tile(tf.expand_dims(eye(self.num_data), 0), [self.num_latent, 1, 1])
        A = I + tf.expand_dims(tf.transpose(self.q_lambda), 1) * \
            tf.expand_dims(tf.transpose(self.q_lambda), 2) * K
        L = tf.cholesky(A)
        Li = tf.matrix_triangular_solve(L, I)
        tmp = Li / tf.expand_dims(tf.transpose(self.q_lambda), 1)
        f_var = 1./tf.square(self.q_lambda) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

        # some statistics about A are used in the KL
        A_logdet = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (A_logdet + trAi - self.num_data * self.num_latent +
                    tf.reduce_sum(K_alpha*self.q_alpha))

        v_exp = self.likelihood.variational_expectations(f_mean, f_var, self.Y)
        return tf.reduce_sum(v_exp) - KL

    def build_predict(self, Xnew, full_cov=False):
        """
        The posterior variance of F is given by

            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)

        Here we project this to F*, the values of the GP at Xnew which is given
        by

           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )

        """

        # compute kernel things
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X)

        # predictive mean
        f_mean = tf.matmul(tf.transpose(Kx), self.q_alpha) + self.mean_function(Xnew)

        # predictive var
        A = K + tf.matrix_diag(tf.transpose(1./tf.square(self.q_lambda)))
        L = tf.cholesky(A)
        Kx_tiled = tf.tile(tf.expand_dims(Kx, 0), [self.num_latent, 1, 1])
        LiKx = tf.matrix_triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kern.K(Xnew) - tf.batch_matmul(LiKx, LiKx, adj_x=True)
        else:
            f_var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
