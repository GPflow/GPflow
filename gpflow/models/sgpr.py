# Copyright 2016 James Hensman, alexggmatthews, Mark van der Wilk
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
from .. import likelihoods
from .. import features

from ..decors import autoflow
from ..decors import params_as_tensors
from ..params import Parameter, DataHolder
from ..mean_functions import Zero

from .model import GPModel

class SGPRUpperMixin(object):
    """
    Upper bound for the GP regression marginal likelihood.
    It is implemented here as a Mixin class which works with SGPR and GPRFITC.
    Note that the same inducing points are used for calculating the upper bound,
    as are used for computing the likelihood approximation. This may not lead to
    the best upper bound. The upper bound can be tightened by optimising Z, just
    as just like the lower bound. This is especially important in FITC, as FITC
    is known to produce poor inducing point locations. An optimisable upper bound
    can be found in https://github.com/markvdw/gp_upper.

    The key reference is

    ::

      @misc{titsias_2014,
        title={Variational Inference for Gaussian and Determinantal Point Processes},
        url={http://www2.aueb.gr/users/mtitsias/papers/titsiasNipsVar14.pdf},
        publisher={Workshop on Advances in Variational Inference (NIPS 2014)},
        author={Titsias, Michalis K.},
        year={2014},
        month={Dec}
      }
    """

    @autoflow()
    @params_as_tensors
    def compute_upper_bound(self):
        num_data = tf.cast(tf.shape(self.Y)[0], settings.float_type)

        Kdiag = self.kern.Kdiag(self.X)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kuf = self.feature.Kuf(self.kern, self.X)

        L = tf.cholesky(Kuu)
        LB = tf.cholesky(Kuu + self.likelihood.variance ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))

        LinvKuf = tf.matrix_triangular_solve(L, Kuf, lower=True)
        # Using the Trace bound, from Titsias' presentation
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(LinvKuf ** 2.0)
        # Kff = self.kern.K(self.X)
        # Qff = tf.matmul(Kuf, LinvKuf, transpose_a=True)

        # Alternative bound on max eigenval:
        # c = tf.reduce_max(tf.reduce_sum(tf.abs(Kff - Qff), 0))
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.log(2 * np.pi * self.likelihood.variance)
        logdet = tf.reduce_sum(tf.log(tf.diag_part(L))) - tf.reduce_sum(tf.log(tf.diag_part(LB)))

        LC = tf.cholesky(Kuu + corrected_noise ** -1.0 * tf.matmul(Kuf, Kuf, transpose_b=True))
        v = tf.matrix_triangular_solve(LC, corrected_noise ** -1.0 * tf.matmul(Kuf, self.Y), lower=True)
        quad = -0.5 * corrected_noise ** -1.0 * tf.reduce_sum(self.Y ** 2.0) + 0.5 * tf.reduce_sum(v ** 2.0)

        return const + logdet + quad


class SGPR(GPModel, SGPRUpperMixin):
    """
    Sparse Variational GP regression. The key reference is

    ::

      @inproceedings{titsias2009variational,
        title={Variational learning of inducing variables in
               sparse Gaussian processes},
        author={Titsias, Michalis K},
        booktitle={International Conference on
                   Artificial Intelligence and Statistics},
        pages={567--574},
        year={2009}
      }



    """

    def __init__(self, X, Y, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.
        """
        X = DataHolder(X)
        Y = DataHolder(Y)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]

    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = len(self.feature)
        num_data = tf.cast(tf.shape(self.Y)[0], settings.float_type)
        output_dim = tf.cast(tf.shape(self.Y)[1], settings.float_type)

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.feature.Kuf(self.kern, self.X)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        L = tf.cholesky(Kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.log(tf.matrix_diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.matrix_diag_part(AAT))

        return bound

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)
        Kuf = self.feature.Kuf(self.kern, self.X)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)
        Kus = self.feature.Kuf(self.kern, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=settings.float_type)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.matmul(tmp1, tmp1, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # P x N x N
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            var = tf.tile(var[:, None], [1, self.num_latent])
        return mean + self.mean_function(Xnew), var


class GPRFITC(GPModel, SGPRUpperMixin):
    def __init__(self, X, Y, kern, feat=None, mean_function=None, Z=None, **kwargs):
        """
        This implements GP regression with the FITC approximation.
        The key reference is

        @inproceedings{Snelson06sparsegaussian,
        author = {Edward Snelson and Zoubin Ghahramani},
        title = {Sparse Gaussian Processes using Pseudo-inputs},
        booktitle = {Advances In Neural Information Processing Systems },
        year = {2006},
        pages = {1257--1264},
        publisher = {MIT press}
        }

        Implementation loosely based on code from GPML matlab library although
        obviously gradients are automatic in GPflow.

        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """

        mean_function = Zero() if mean_function is None else mean_function

        X = DataHolder(X)
        Y = DataHolder(Y)
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.feature = features.inducingpoint_wrapper(feat, Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    @params_as_tensors
    def _build_common_terms(self):
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)  # size N x R
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.feature.Kuf(self.kern, self.X)
        Kuu = self.feature.Kuu(self.kern, jitter=settings.numerics.jitter_level)

        Luu = tf.cholesky(Kuu)  # => Luu Luu^T = Kuu
        V = tf.matrix_triangular_solve(Luu, Kuf)  # => V^T V = Qff = Kuf^T Kuu^-1 Kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = tf.eye(num_inducing, dtype=settings.float_type) + tf.matmul(V / nu, V, transpose_b=True)
        L = tf.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size N x R
        alpha = tf.matmul(V, beta)  # size N x R

        gamma = tf.matrix_triangular_solve(L, alpha, lower=True)  # size N x R

        return err, nu, Luu, L, alpha, beta, gamma

    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu Kuu^{-1} Kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #            = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #            = \diag( \nu^{-1} ) - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1) V \diag(\nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* ( \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha ) )

        err, nu, Luu, L, alpha, beta, gamma = self._build_common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(tf.square(err) / tf.expand_dims(nu, 1)) \
                          + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * tf.log(tf.constant(2. * np.pi, settings.float_type))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.log(nu)) - tf.reduce_sum(tf.log(tf.matrix_diag_part(L)))
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self._build_common_terms()
        Kus = self.feature.Kuf(self.kern, Xnew)  # size  M x Xnew

        w = tf.matrix_triangular_solve(Luu, Kus, lower=True)  # size M x Xnew

        tmp = tf.matrix_triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.matrix_triangular_solve(L, w, lower=True)

        if full_cov:
            var = self.kern.K(Xnew) - tf.matmul(w, w, transpose_a=True) \
                  + tf.matmul(intermediateA, intermediateA, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # P x N x N
        else:
            var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(w), 0) \
                  + tf.reduce_sum(tf.square(intermediateA), 0)  # size Xnew,
            var = tf.tile(var[:, None], [1, self.num_latent])

        return mean, var

    @property
    def Z(self):
        raise NotImplementedError("Inducing points are now in `model.feature.Z`.")

    @Z.setter
    def Z(self, _):
        raise NotImplementedError("Inducing points are now in `model.feature.Z`.")
