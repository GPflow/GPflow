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
def gauss_kl_white(q_mu, q_sqrt):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance.
    """
    KL = 0.5 * tf.reduce_sum(tf.square(q_mu))  # Mahalanobis term
    KL += -0.5 * tf.cast(tf.reduce_prod(tf.shape(q_sqrt)[1:]), float_type)  # constant term
    L = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
    KL -= 0.5 * tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(L))))  # logdet
    KL += 0.5 * tf.reduce_sum(tf.square(L))  # Trace term.
    return KL


@NameScoped("KL")
def gauss_kl_white_diag(q_mu, q_sqrt):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, I)

    We assume multiple independent distributions, given by the columns of
    q_mu and q_sqrt

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance.
    """

    KL = 0.5 * tf.reduce_sum(tf.square(q_mu))  # Mahalanobis term
    KL += -0.5 * tf.cast(tf.size(q_sqrt), float_type)
    KL += -0.5 * tf.reduce_sum(tf.log(tf.square(q_sqrt)))  # Log-det of q-cov
    KL += 0.5 * tf.reduce_sum(tf.square(q_sqrt))  # Trace term
    return KL


@NameScoped("KL")
def gauss_kl_diag(q_mu, q_sqrt, K):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume multiple independent distributions, given by the columns of
    q_mu and q_sqrt.

    q_mu is a matrix, each column contains a mean

    q_sqrt is a matrix, each column represents the diagonal of a square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    """
    L = tf.cholesky(K)
    alpha = tf.matrix_triangular_solve(L, q_mu, lower=True)
    KL = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    num_latent = tf.cast(tf.shape(q_sqrt)[1], float_type)
    KL += num_latent * 0.5 * tf.reduce_sum(
        tf.log(tf.square(tf.matrix_diag_part(L))))  # Prior log-det term.
    KL += -0.5 * tf.cast(tf.size(q_sqrt), float_type)  # constant term
    KL += -0.5 * tf.reduce_sum(tf.log(tf.square(q_sqrt)))  # Log-det of q-cov
    L_inv = tf.matrix_triangular_solve(L, tf.eye(tf.shape(L)[0], dtype=float_type), lower=True)
    K_inv = tf.matrix_triangular_solve(tf.transpose(L), L_inv, lower=False)
    KL += 0.5 * tf.reduce_sum(tf.expand_dims(tf.matrix_diag_part(K_inv), 1) *
                              tf.square(q_sqrt))  # Trace term.
    return KL


@NameScoped("KL")
def gauss_kl(q_mu, q_sqrt, K):
    """
    Compute the KL divergence from

          q(x) = N(q_mu, q_sqrt^2)
    to
          p(x) = N(0, K)

    We assume multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt.

    q_mu is a matrix, each column contains a mean.

    q_sqrt is a 3D tensor, each matrix within is a lower triangular square-root
        matrix of the covariance of q.

    K is a positive definite matrix: the covariance of p.
    """
    L = tf.cholesky(K)
    alpha = tf.matrix_triangular_solve(L, q_mu, lower=True)
    KL = 0.5 * tf.reduce_sum(tf.square(alpha))  # Mahalanobis term.
    num_latent = tf.cast(tf.shape(q_sqrt)[2], float_type)
    KL += num_latent * 0.5 * tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(L))))  # Prior log-det term.
    KL += -0.5 * tf.cast(tf.reduce_prod(tf.shape(q_sqrt)[1:]), float_type)  # constant term
    Lq = tf.matrix_band_part(tf.transpose(q_sqrt, (2, 0, 1)), -1, 0)  # force lower triangle
    KL += -0.5*tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(Lq))))  # logdet
    L_tiled = tf.tile(tf.expand_dims(L, 0), tf.stack([tf.shape(Lq)[0], 1, 1]))
    LiLq = tf.matrix_triangular_solve(L_tiled, Lq, lower=True)
    KL += 0.5 * tf.reduce_sum(tf.square(LiLq))  # Trace term
    return KL
