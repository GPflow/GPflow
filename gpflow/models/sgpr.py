# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

import numpy as np
import tensorflow as tf

from gpflow.kernels import Kernel

from .. import likelihoods
from ..config import default_float, default_jitter
from ..covariances.dispatch import Kuf, Kuu
from ..inducing_variables import InducingPoints
from ..mean_functions import MeanFunction
from ..utilities import to_default_float
from .model import GPModel, MeanAndVariance
from .training_mixins import InputData, InternalDataTrainingLossMixin, RegressionData
from .util import data_input_to_tensor, inducingpoint_wrapper


class SGPRBase(GPModel, InternalDataTrainingLossMixin):
    """
    Common base class for SGPR and GPRFITC that provides the common __init__
    and upper_bound() methods.
    """

    def __init__(
        self,
        data: RegressionData,
        kernel: Kernel,
        inducing_variable: InducingPoints,
        *,
        mean_function: Optional[MeanFunction] = None,
        num_latent_gps: Optional[int] = None,
        noise_variance: float = 1.0,
    ):
        """
        `data`:  a tuple of (X, Y), where the inputs X has shape [N, D]
            and the outputs Y has shape [N, R].
        `inducing_variable`:  an InducingPoints instance or a matrix of
            the pseudo inputs Z, of shape [M, D].
        `kernel`, `mean_function` are appropriate GPflow objects

        This method only works with a Gaussian likelihood, its variance is
        initialized to `noise_variance`.
        """
        likelihood = likelihoods.Gaussian(noise_variance)
        X_data, Y_data = data_input_to_tensor(data)
        num_latent_gps = Y_data.shape[-1] if num_latent_gps is None else num_latent_gps
        super().__init__(kernel, likelihood, mean_function, num_latent_gps=num_latent_gps)

        self.data = X_data, Y_data
        self.num_data = X_data.shape[0]

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

    def upper_bound(self) -> tf.Tensor:
        """
        Upper bound for the sparse GP regression marginal likelihood.  Note that
        the same inducing points are used for calculating the upper bound, as are
        used for computing the likelihood approximation. This may not lead to the
        best upper bound. The upper bound can be tightened by optimising Z, just
        like the lower bound. This is especially important in FITC, as FITC is
        known to produce poor inducing point locations. An optimisable upper bound
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

        The key quantity, the trace term, can be computed via

        >>> _, v = conditionals.conditional(X, model.inducing_variable.Z, model.kernel,
        ...                                 np.zeros((model.inducing_variable.num_inducing, 1)))

        which computes each individual element of the trace term.
        """
        X_data, Y_data = self.data
        num_data = to_default_float(tf.shape(Y_data)[0])

        Kdiag = self.kernel(X_data, full_cov=False)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)

        I = tf.eye(tf.shape(kuu)[0], dtype=default_float())

        L = tf.linalg.cholesky(kuu)
        A = tf.linalg.triangular_solve(L, kuf, lower=True)
        AAT = tf.linalg.matmul(A, A, transpose_b=True)
        B = I + AAT / self.likelihood.variance
        LB = tf.linalg.cholesky(B)

        # Using the Trace bound, from Titsias' presentation
        c = tf.reduce_sum(Kdiag) - tf.reduce_sum(tf.square(A))

        # Alternative bound on max eigenval:
        corrected_noise = self.likelihood.variance + c

        const = -0.5 * num_data * tf.math.log(2 * np.pi * self.likelihood.variance)
        logdet = -tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))

        err = Y_data - self.mean_function(X_data)
        LC = tf.linalg.cholesky(I + AAT / corrected_noise)
        v = tf.linalg.triangular_solve(LC, tf.linalg.matmul(A, err) / corrected_noise, lower=True)
        quad = -0.5 * tf.reduce_sum(tf.square(err)) / corrected_noise + 0.5 * tf.reduce_sum(
            tf.square(v)
        )

        return const + logdet + quad


class SGPR(SGPRBase):
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

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.elbo()

    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """
        X_data, Y_data = self.data

        num_inducing = self.inducing_variable.num_inducing
        num_data = to_default_float(tf.shape(Y_data)[0])
        output_dim = to_default_float(tf.shape(Y_data)[1])

        err = Y_data - self.mean_function(X_data)
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
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

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)
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
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])
        return mean + self.mean_function(Xnew), var

    def compute_qu(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Computes the mean and variance of q(u) = N(mu, cov), the variational distribution on
        inducing outputs. SVGP with this q(u) should predict identically to
        SGPR.
        :return: mu, cov
        """
        X_data, Y_data = self.data

        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        sig = kuu + (self.likelihood.variance ** -1) * tf.matmul(kuf, kuf, transpose_b=True)
        sig_sqrt = tf.linalg.cholesky(sig)

        sig_sqrt_kuu = tf.linalg.triangular_solve(sig_sqrt, kuu)

        cov = tf.linalg.matmul(sig_sqrt_kuu, sig_sqrt_kuu, transpose_a=True)
        err = Y_data - self.mean_function(X_data)
        mu = (
            tf.linalg.matmul(
                sig_sqrt_kuu,
                tf.linalg.triangular_solve(sig_sqrt, tf.linalg.matmul(kuf, err)),
                transpose_a=True,
            )
            / self.likelihood.variance
        )

        return mu, cov


class GPRFITC(SGPRBase):
    """
    This implements GP regression with the FITC approximation.
    The key reference is

    ::

      @inproceedings{Snelson06sparsegaussian,
        author = {Edward Snelson and Zoubin Ghahramani},
        title = {Sparse Gaussian Processes using Pseudo-inputs},
        booktitle = {Advances In Neural Information Processing Systems},
        year = {2006},
        pages = {1257--1264},
        publisher = {MIT press}
      }

    Implementation loosely based on code from GPML matlab library although
    obviously gradients are automatic in GPflow.
    """

    def common_terms(self):
        X_data, Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        err = Y_data - self.mean_function(X_data)  # size [N, R]
        Kdiag = self.kernel(X_data, full_cov=False)
        kuf = Kuf(self.inducing_variable, self.kernel, X_data)
        kuu = Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())

        Luu = tf.linalg.cholesky(kuu)  # => Luu Luu^T = kuu
        V = tf.linalg.triangular_solve(Luu, kuf)  # => V^T V = Qff = kuf^T kuu^-1 kuf

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = tf.eye(num_inducing, dtype=default_float()) + tf.linalg.matmul(
            V / nu, V, transpose_b=True
        )
        L = tf.linalg.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size [N, R]
        alpha = tf.linalg.matmul(V, beta)  # size [N, R]

        gamma = tf.linalg.triangular_solve(L, alpha, lower=True)  # size [N, R]

        return err, nu, Luu, L, alpha, beta, gamma

    def maximum_log_likelihood_objective(self) -> tf.Tensor:
        return self.fitc_log_marginal_likelihood()

    def fitc_log_marginal_likelihood(self) -> tf.Tensor:
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

        mahalanobisTerm = -0.5 * tf.reduce_sum(
            tf.square(err) / tf.expand_dims(nu, 1)
        ) + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * tf.math.log(tf.constant(2.0 * np.pi, default_float()))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.math.log(nu)) - tf.reduce_sum(
            tf.math.log(tf.linalg.diag_part(L))
        )
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm * self.num_latent_gps

    def predict_f(self, Xnew: InputData, full_cov=False, full_output_cov=False) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self.common_terms()
        Kus = Kuf(self.inducing_variable, self.kernel, Xnew)  # [M, N]

        w = tf.linalg.triangular_solve(Luu, Kus, lower=True)  # [M, N]

        tmp = tf.linalg.triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.linalg.matmul(w, tmp, transpose_a=True) + self.mean_function(Xnew)
        intermediateA = tf.linalg.triangular_solve(L, w, lower=True)

        if full_cov:
            var = (
                self.kernel(Xnew)
                - tf.linalg.matmul(w, w, transpose_a=True)
                + tf.linalg.matmul(intermediateA, intermediateA, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                - tf.reduce_sum(tf.square(w), 0)
                + tf.reduce_sum(tf.square(intermediateA), 0)
            )  # [N, P]
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        return mean, var
