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

    q_mu is a matrix (M x L), each column contains a mean.

    q_sqrt can be a 3D tensor (L xM x M), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix (M x L), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    K is the covariance of p.
    It is a positive definite matrix (M x M) or a tensor of stacked such matrices (L x M x M)
    If K is None, compute the KL divergence to p(x) = N(0, I) instead.
    """

    # TODO: why do I need num_latents if I already have L (object type problem)
    white = True if K is None else False
    diag = True if q_sqrt.get_shape().ndims == 2 else False

    if white:
        alpha = q_mu  # M x L
    else:
        batch = True if K.get_shape().ndims == 3 else False
        Lp = tf.cholesky(K)  # L x M x M or M x M
        q_mu = tf.transpose(q_mu)[:, :, None] if K.shape.ndims == 3 else q_mu  # L x M x 1 or M x L
        alpha = tf.matrix_triangular_solve(Lp, q_mu, lower=True)  # L x M x 1 or M x L

    if diag:
        num_latent = tf.shape(q_sqrt)[1]
        Lq_diag = q_sqrt  # M x L
        # one has to invert Lp anyway so little is wasted in expanding q_sqrt
        Lq = tf.matrix_diag(tf.transpose(q_sqrt)) # L x M x M
    else:
        num_latent = tf.shape(q_sqrt)[0]
        Lq = tf.matrix_band_part(q_sqrt, -1, 0)  # force lower triangle # L x M x M
        Lq_diag = tf.matrix_diag_part(Lq) # # M x L

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - N x M
    constant = - tf.size(q_mu, out_type=settings.float_type)  # ML

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        # If K is L x M x M, no need to tile the cholesky anymore
        # TODO: broadcast instead of tile when tf allows (not implemented in tf <= 1.6.0)
        Lp_tiled = Lp if batch else tf.tile(tf.expand_dims(Lp, 0), [num_latent, 1, 1])
        LpiLq = tf.matrix_triangular_solve(Lp_tiled, Lq, lower=True)
        trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.log(tf.square(tf.matrix_diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is L x M x M, num_latent is no longer implicit, no need to multiply the single kernel logdet
        scale = tf.cast(num_latent, settings.float_type) if K.get_shape().ndims == 2 else 1.0
        prior_logdet = scale * sum_log_sqdiag_Lp
        twoKL += prior_logdet

    return 0.5 * twoKL