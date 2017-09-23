# Copyright 2016 James Hensman, alexggmatthews
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
from .scoping import NameScoped
from ._settings import settings
float_type = settings.dtypes.float_type


@NameScoped("KL")
def gauss_kl(q_mu, q_sqrt, K=None):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean.

    q_sqrt can be a 3D tensor, each matrix within is a lower triangular
        square-root matrix of the covariance of q.
    q_sqrt can be a matrix, each column represents the diagonal of a square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    If K is None, compute the KL divergence to p(x) = N(0, I) instead.

    These functions are now considered deprecated, subsumed into this one:
        gauss_kl_white
        gauss_kl_white_diag
        gauss_kl_diag
    """
    if K is None:
        white = True
        alpha = q_mu
    else:
        white = False
        Lp = tf.cholesky(K)
        alpha = tf.matrix_triangular_solve(Lp, q_mu, lower=True)

    if q_sqrt.get_shape().ndims == 2:
        diag = True
        num_latent = tf.cast(tf.shape(q_sqrt)[1], float_type)
        NM = tf.size(q_sqrt)
        Lq = Lq_diag = q_sqrt
    elif q_sqrt.get_shape().ndims == 3:
        diag = False
        num_latent = tf.cast(tf.shape(q_sqrt)[2], float_type)
        NM = tf.reduce_prod(tf.shape(q_sqrt)[1:])
        Lq = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
        Lq_diag = tf.matrix_diag_part(Lq)
    else: # pragma: no cover
        raise ValueError("Bad dimension for q_sqrt: %s" %
                         str(q_sqrt.get_shape().ndims))

    mahalanobis = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    
    constant = -0.5 * tf.cast(NM, float_type)  # constant term
    logdet_qcov = -0.5 * tf.reduce_sum(tf.log(tf.square(Lq_diag)))  # Log-det of q-cov

    if white:
        trace = 0.5 * tf.reduce_sum(tf.square(Lq))  # Trace term.
    else:
        if diag:
            Lp_inv = tf.matrix_triangular_solve(Lp, tf.eye(tf.shape(Lp)[0], dtype=float_type), lower=True)
            K_inv = tf.matrix_triangular_solve(tf.transpose(Lp), Lp_inv, lower=False)
            trace = 0.5 * tf.reduce_sum(tf.expand_dims(tf.matrix_diag_part(K_inv), 1) *
                                      tf.square(q_sqrt))  # Trace term.
        else:
            Lp_tiled = tf.tile(tf.expand_dims(Lp, 0), tf.stack([tf.shape(Lq)[0], 1, 1]))
            LpiLq = tf.matrix_triangular_solve(Lp_tiled, Lq, lower=True)
            trace = 0.5 * tf.reduce_sum(tf.square(LpiLq))  # Trace term

    KL = mahalanobis + constant + logdet_qcov + trace

    if not white:
        prior_logdet = num_latent * 0.5 * tf.reduce_sum(
            tf.log(tf.square(tf.matrix_diag_part(Lp))))  # Prior log-det term.
        KL += prior_logdet

    return KL


import warnings


def gauss_kl_white(q_mu, q_sqrt):
    warnings.warn('gauss_kl_white is deprecated: use gauss_kl(...) instead',
                  DeprecationWarning)
    return gauss_kl(q_mu, q_sqrt)


def gauss_kl_white_diag(q_mu, q_sqrt):
    warnings.warn('gauss_kl_white_diag is deprecated: use gauss_kl(...) instead',
                  DeprecationWarning)
    return gauss_kl(q_mu, q_sqrt)


def gauss_kl_diag(q_mu, q_sqrt, K):
    warnings.warn('gauss_kl_diag is deprecated: use gauss_kl(...) instead',
                  DeprecationWarning)
    return gauss_kl(q_mu, q_sqrt, K)
