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
import numpy as np

from .. import settings
from .. import transforms

from ..params import Parameter
from ..params import DataHolder
from ..decors import params_as_tensors

from ..mean_functions import Zero
from ..conditionals import conditional
from ..kullback_leiblers import gauss_kl
from ..models.model import GPModel


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
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        """

        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, num_latent, **kwargs)
        self.num_data = X.shape[0]

        self.q_mu = Parameter(np.zeros((self.num_data, self.num_latent)))
        q_sqrt = np.array([np.eye(self.num_data)
                           for _ in range(self.num_latent)])
        transform = transforms.LowerTriangular(self.num_data, self.num_latent)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)

    def compile(self, session=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_mu = Parameter(np.zeros((self.num_data, self.num_latent)))
            self.q_sqrt = Parameter(np.eye(self.num_data)[:, :, None] *
                                    np.ones((1, 1, self.num_latent)))

        return super(VGP, self).compile(session=session)

    @params_as_tensors
    def _build_likelihood(self):
        """
        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(\\mathbf f) = N(\\mathbf f \\,|\\, \\boldsymbol \\mu, \\boldsymbol \\Sigma)

        """

        # Get prior KL.
        KL = gauss_kl(self.q_mu, self.q_sqrt)

        # Get conditionals
        K = self.kern.K(self.X) + tf.eye(self.num_data, dtype=settings.float_type) * \
            settings.numerics.jitter_level
        L = tf.cholesky(K)

        fmean = tf.matmul(L, self.q_mu) + self.mean_function(self.X)  # NN,ND->ND

        q_sqrt_dnn = tf.matrix_band_part(self.q_sqrt, -1, 0)  # D x N x N

        L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([self.num_latent, 1, 1]))

        LTA = tf.matmul(L_tiled, q_sqrt_dnn)  # D x N x N
        fvar = tf.reduce_sum(tf.square(LTA), 2)

        fvar = tf.transpose(fvar)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        return tf.reduce_sum(var_exp) - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        mu, var = conditional(Xnew, self.X, self.kern, self.q_mu,
                              q_sqrt=self.q_sqrt, full_cov=full_cov, white=True)
        return mu + self.mean_function(Xnew), var


class VGP_opper_archambeau(GPModel):
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
       q(\\mathbf f) = N(\\mathbf f \\,|\\, \\mathbf K \\boldsymbol \\alpha,
                         [\\mathbf K^{-1} + \\textrm{diag}(\\boldsymbol \\lambda))^2]^{-1})

    This approach has only 2ND parameters, rather than the N + N^2 of vgp,
    but the optimization is non-convex and in practice may cause difficulty.

    """

    def __init__(self, X, Y, kern, likelihood,
                 mean_function=None,
                 num_latent=None,
                 **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects
        """

        mean_function = Zero() if mean_function is None else mean_function

        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.q_alpha = Parameter(np.zeros((self.num_data, self.num_latent)))
        self.q_lambda = Parameter(np.ones((self.num_data, self.num_latent)),
                                  transforms.positive)

    def compile(self, session=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the shape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_alpha = Parameter(np.zeros((self.num_data, self.num_latent)))
            self.q_lambda = Parameter(np.ones((self.num_data, self.num_latent)),
                                      transforms.positive)
        return super(VGP_opper_archambeau, self).compile(session=session)

    @params_as_tensors
    def _build_likelihood(self):
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
        I = tf.tile(tf.expand_dims(tf.eye(self.num_data, dtype=settings.float_type), 0),
                    [self.num_latent, 1, 1])
        A = I + tf.expand_dims(tf.transpose(self.q_lambda), 1) * \
            tf.expand_dims(tf.transpose(self.q_lambda), 2) * K
        L = tf.cholesky(A)
        Li = tf.matrix_triangular_solve(L, I)
        tmp = Li / tf.expand_dims(tf.transpose(self.q_lambda), 1)
        f_var = 1. / tf.square(self.q_lambda) - tf.transpose(tf.reduce_sum(tf.square(tmp), 1))

        # some statistics about A are used in the KL
        A_logdet = 2.0 * tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        trAi = tf.reduce_sum(tf.square(Li))

        KL = 0.5 * (A_logdet + trAi - self.num_data * self.num_latent +
                    tf.reduce_sum(K_alpha * self.q_alpha))

        v_exp = self.likelihood.variational_expectations(f_mean, f_var, self.Y)
        return tf.reduce_sum(v_exp) - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
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
        f_mean = tf.matmul(Kx, self.q_alpha, transpose_a=True) + self.mean_function(Xnew)

        # predictive var
        A = K + tf.matrix_diag(tf.transpose(1. / tf.square(self.q_lambda)))
        L = tf.cholesky(A)
        Kx_tiled = tf.tile(tf.expand_dims(Kx, 0), [self.num_latent, 1, 1])
        LiKx = tf.matrix_triangular_solve(L, Kx_tiled)
        if full_cov:
            f_var = self.kern.K(Xnew) - tf.matmul(LiKx, LiKx, transpose_a=True)
        else:
            f_var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(LiKx), 1)
        return f_mean, tf.transpose(f_var)
