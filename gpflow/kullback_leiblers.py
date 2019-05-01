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
from .util import default_float


def gauss_kl(q_mu, q_sqrt, K=None):
    """
    Compute the KL divergence KL[q || p] between

          q(x) = N(q_mu, q_sqrt^2)
    and
          p(x) = N(0, K)

    We assume N multiple independent distributions, given by the columns of
    q_mu and the last dimension of q_sqrt. Returns the sum of the divergences.

    q_mu is a matrix ([M, L]), each column contains a mean.

    q_sqrt can be a 3D tensor ([L, M, M]), each matrix within is a lower
        triangular square-root matrix of the covariance of q.
    q_sqrt can be a matrix ([M, L]), each column represents the diagonal of a
        square-root matrix of the covariance of q.

    K is the covariance of p.
    It is a positive definite matrix ([M, M]) or a tensor of stacked such matrices ([L, M, M])
    If K is None, compute the KL divergence to p(x) = N(0, I) instead.
    """

    white = K is None
    diag = q_sqrt.shape.ndims == 2

    M, B = q_mu.shape[0], q_mu.shape[1]

    if white:
        alpha = q_mu  # [M, B]
    else:
        batch = K.shape.ndims == 3

        Lp = tf.linalg.cholesky(K)  # [B, M, M] or [M, M]
        q_mu = tf.transpose(
            q_mu)[:, :, None] if batch else q_mu  # [B, M, 1] or [M, B]
        alpha = tf.linalg.triangular_solve(Lp, q_mu,
                                           lower=True)  # [B, M, 1] or [M, B]

    if diag:
        Lq = Lq_diag = q_sqrt
        Lq_full = tf.linalg.diag(tf.transpose(q_sqrt))  # [B, M, M]
    else:
        Lq = Lq_full = tf.linalg.band_part(
            q_sqrt, -1, 0)  # force lower triangle # [B, M, M]
        Lq_diag = tf.linalg.diag_part(Lq)  # [M, B]

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - B * M
    constant = -tf.cast(tf.size(q_mu, out_type=tf.int64),
                        dtype=default_float())

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if diag and not batch:
            # K is [M, M] and q_sqrt is [M, B]: fast specialisation
            LpT = tf.transpose(Lp)  # [M, M]
            Lp_inv = tf.linalg.triangular_solve(Lp,
                                                tf.eye(M,
                                                       dtype=default_float()),
                                                lower=True)  # [M, M]
            K_inv = tf.linalg.diag_part(
                tf.linalg.triangular_solve(
                    LpT, Lp_inv, lower=False))[:, None]  # [M, M] -> [M, 1]
            trace = tf.reduce_sum(K_inv * tf.square(q_sqrt))
        else:
            # TODO: broadcast instead of tile when tf allows (not implemented in tf <= 1.6.0)
            Lp_full = Lp if batch else tf.tile(tf.expand_dims(Lp, 0),
                                               [B, 1, 1])
            LpiLq = tf.linalg.triangular_solve(Lp_full, Lq_full, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not white:
        log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is [B, M, M], num_latent is no longer implicit, no need to multiply the single kernel logdet
        scale = 1.0 if batch else tf.cast(B, default_float())
        twoKL += scale * sum_log_sqdiag_Lp

    return 0.5 * twoKL
