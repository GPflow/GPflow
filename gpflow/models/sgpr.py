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


import numpy as np
import tensorflow as tf

from gpflow.features import InducingPoints
from gpflow.covariances.dispatch import Kuf, Kuu
from gpflow.util import default_float, default_jitter

from .model import GPModel, GPModelOLD, MeanAndVariance
from .. import features
from .. import likelihoods
from ..mean_functions import Zero


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

    def upper_bound(self):
        num_data = tf.cast(tf.shape(self.Y)[0], default_float())

        Kdiag = self.kernel(self.X)
        kuu = Kuu(self.feature, self.kernel, jitter=default_jitter())
        kuf = Kuf(self.feature, self.kernel, self.X)

        L = tf.linalg.cholesky(kuu)
        LB = tf.linalg.cholesky(
            kuu + self.likelihood.variance ** -1.0 * tf.linalg.matmul(kuf, kuf, transpose_b=True))

        Linvkuf = tf.linalg.triangular_solve(L, kuf, lower=True)
        # Using the Trace bound, from Titsias' presentation
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(Linvkuf ** 2.0)
        # Kff = self.kernel(self.X)
        # Qff = tf.linalg.matmul(kuf, Linvkuf, transpose_a=True)

        # Alternative bound on max eigenval:
        # c = tf.reduce_max(tf.reduce_sum(tf.abs(Kff - Qff), 0))
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.math.log(2 * np.pi * self.likelihood.variance)
        logdet = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L))) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(LB)))

        LC = tf.linalg.cholesky(
            kuu + corrected_noise ** -1.0 * tf.linalg.matmul(kuf, kuf, transpose_b=True))
        v = tf.linalg.triangular_solve(LC, corrected_noise ** -1.0 * tf.linalg.matmul(kuf, self.Y),
                                       lower=True)
        quad = -0.5 * corrected_noise ** -1.0 * tf.reduce_sum(self.Y ** 2.0) + 0.5 * tf.reduce_sum(
            v ** 2.0)

        return const + logdet + quad


class SGPR(GPModelOLD, SGPRUpperMixin):
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

    def __init__(self, X, Y, kernel, mean_function=None, features=None, **kwargs):
        """
        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        Z is a matrix of pseudo inputs, size [M, D]
        kernel, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.
        """
        likelihood = likelihoods.Gaussian()
        GPModelOLD.__init__(self, X, Y, kernel, likelihood, mean_function, **kwargs)
        self.feature = InducingPoints(features)
        self.num_data = X.shape[0]

    def log_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = len(self.feature)
        num_data = tf.cast(tf.shape(self.Y)[0], default_float())
        output_dim = tf.cast(tf.shape(self.Y)[1], default_float())

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kernel(self.X)
        kuf = Kuf(self.feature, self.kernel, self.X)
        kuu = Kuu(self.feature, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += tf.negative(output_dim) * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.math.log(self.likelihood.variance)
        bound += -0.5 * tf.reduce_sum(tf.square(err)) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * output_dim * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * output_dim * tf.reduce_sum(tf.linalg.diag_part(AAT))

        return bound

    def predict_f(self, X: tf.Tensor, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)
        kuf = Kuf(self.feature, self.kernel, self.X)
        kuu = Kuu(self.feature, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.feature, self.kernel, X)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True) / sigma
        B = tf.linalg.matmul(A, A, transpose_b=True) + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        Aerr = tf.linalg.matmul(A, err)
        c = tf.linalg.triangular_solve(LB, Aerr, lower=True) / sigma
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = self.kernel(X) + tf.linalg.matmul(tmp2, tmp2, transpose_a=True) \
                  - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # [P, N, N]
        else:
            var = self.kernel(X, full=False) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            var = tf.tile(var[:, None], [1, self.num_latent])
        return mean + self.mean_function(X), var


class GPRFITC(GPModelOLD, SGPRUpperMixin):
    def __init__(self, X, Y, kernel, mean_function=None, features=None, **kwargs):
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

        X is a data matrix, size [N, D]
        Y is a data matrix, size [N, R]
        Z is a matrix of pseudo inputs, size [M, D]
        kernel, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """

        mean_function = Zero() if mean_function is None else mean_function

        likelihood = likelihoods.Gaussian()
        GPModelOLD.__init__(self, X, Y, kernel, likelihood, mean_function, **kwargs)
        self.feature = InducingPoints(features)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def common_terms(self):
        num_inducing = len(self.feature)
        err = self.Y - self.mean_function(self.X)  # size [N, R]
        Kdiag = self.kernel(self.X, full=False)
        kuf = Kuf(self.feature, self.kernel, self.X)
        kuu = Kuu(self.feature, self.kernel, jitter=default_jitter())

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(V / nu, V,
                                                                           transpose_b=True)
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [N, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]

        return err, nu, Luu, L, alpha, beta, gamma

    def log_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu kuu^{-1} kuf
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

        err, nu, Luu, L, alpha, beta, gamma = self.common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(tf.square(err) / tf.expand_dims(nu, 1)) \
                          + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2. * np.pi, default_float()))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L)))
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent

    def predict_f(self, X: tf.Tensor, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self.common_terms()
        Kus = Kuf(self.feature, self.kernel, X)  # size  [M, X]new

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # size [M, X]new

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(X)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = self.kernel(X) - tf.linalg.matmul(w, w, transpose_a=True) \
                  + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            var = tf.tile(var[None, ...], [self.num_latent, 1, 1])  # [P, N, N]
        else:
            var = self.kernel(X, full=False) - tf.reduce_sum(tf.square(w), 0) \
                  + tf.reduce_sum(tf.square(intermediateA), 0)  # size Xnew,
            var = tf.tile(var[:, None], [1, self.num_latent])

        return mean, var

    @property
    def Z(self):
        raise NotImplementedError("Inducing points are now in `model.feature.Z()`.")

    @Z.setter
    def Z(self, _):
        raise NotImplementedError("Inducing points are now in `model.feature.Z()`.")
