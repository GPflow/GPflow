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

# -*- coding: utf-8 -*-


import tensorflow as tf

from . import settings
from .decors import name_scope


@name_scope()
def gauss_kl(q_mu, q_sqrt, K=None):
    """
    Compute the KL divergence KL[q || p] between

          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(0, K)

    We assume N multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt. Returns the sum of the divergences.

    q_mu is a matrix (M x N), each column contains a mean.

    q_sqrt can be a 3D tensor (N xM x M), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix (M x N), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    K is a positive definite matrix (M x M): the covariance of p.
    If K is None, compute the KL divergence to p(x) = N(0, I) instead.
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
        num_latent = tf.shape(q_sqrt)[1]
        NM = tf.size(q_sqrt)
        Lq = Lq_diag = q_sqrt
    elif q_sqrt.get_shape().ndims == 3:
        diag = False
        num_latent = tf.shape(q_sqrt)[0]
        NM = tf.reduce_prod(tf.shape(q_sqrt)[:2])
        Lq = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle
        Lq_diag = tf.matrix_diag_part(Lq)
    else:  # pragma: no cover
        raise ValueError("Bad dimension for q_sqrt: {}".format(q_sqrt.get_shape().ndims))

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - N x M
    constant = - tf.cast(NM, settings.float_type)

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if diag:
            M = tf.shape(Lp)[0]
            Lp_inv = tf.matrix_triangular_solve(
                Lp, tf.eye(M, dtype=settings.float_type), lower=True)
            K_inv = tf.matrix_triangular_solve(
                tf.transpose(Lp), Lp_inv, lower=False)
            trace = tf.reduce_sum(
                tf.expand_dims(tf.matrix_diag_part(K_inv), 1) * tf.square(q_sqrt))
        else:
            Lp_tiled = tf.tile(tf.expand_dims(Lp, 0), [num_latent, 1, 1])
            LpiLq = tf.matrix_triangular_solve(Lp_tiled, Lq, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        prior_logdet = tf.cast(num_latent, settings.float_type) * sum_log_sqdiag_Lp
        twoKL += prior_logdet

    return 0.5 * twoKL

