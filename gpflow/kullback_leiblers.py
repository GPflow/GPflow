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

# -*- coding: utf-8 -*-

import tensorflow as tf
from packaging.version import Version

from .base import TensorType
from .config import default_float, default_jitter
from .covariances import Kuu
from .experimental.check_shapes import check_shapes
from .inducing_variables import InducingVariables
from .kernels import Kernel
from .utilities import Dispatcher, to_default_float

prior_kl = Dispatcher("prior_kl")


@prior_kl.register(InducingVariables, Kernel, object, object)
@check_shapes(
    "inducing_variable: [N, D, broadcast L]",
    "q_mu: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return: []",
)
def _(
    inducing_variable: InducingVariables,
    kernel: Kernel,
    q_mu: TensorType,
    q_sqrt: TensorType,
    whiten: bool = False,
) -> tf.Tensor:
    if whiten:
        return gauss_kl(q_mu, q_sqrt, None)
    else:
        K = Kuu(inducing_variable, kernel, jitter=default_jitter())  # [P, M, M] or [M, M]
        return gauss_kl(q_mu, q_sqrt, K)


@check_shapes(
    "q_mu: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "K: [broadcast L, M, M]",
    "K_cholesky: [broadcast L, M, M]",
    "return: []",
)
def gauss_kl(
    q_mu: TensorType, q_sqrt: TensorType, K: TensorType = None, *, K_cholesky: TensorType = None
) -> tf.Tensor:
    """
    Compute the KL divergence KL[q || p] between::

          q(x) = N(q_mu, q_sqrt^2)

    and::

          p(x) = N(0, K)    if K is not None
          p(x) = N(0, I)    if K is None

    We assume L multiple independent distributions, given by the columns of
    q_mu and the first or last dimension of q_sqrt. Returns the *sum* of the
    divergences.

    q_mu is a matrix ([M, L]), each column contains a mean.

    - q_sqrt can be a 3D tensor ([L, M, M]), each matrix within is a lower
      triangular square-root matrix of the covariance of q.
    - q_sqrt can be a matrix ([M, L]), each column represents the diagonal of a
      square-root matrix of the covariance of q.

    K is the covariance of p (positive-definite matrix).  The K matrix can be
    passed either directly as `K`, or as its Cholesky factor, `K_cholesky`.  In
    either case, it can be a single matrix [M, M], in which case the sum of the
    L KL divergences is computed by broadcasting, or L different covariances
    [L, M, M].

    Note: if no K matrix is given (both `K` and `K_cholesky` are None),
    `gauss_kl` computes the KL divergence from p(x) = N(0, I) instead.
    """

    if (K is not None) and (K_cholesky is not None):
        raise ValueError(
            "Ambiguous arguments: gauss_kl() must only be passed one of `K` or `K_cholesky`."
        )

    is_white = (K is None) and (K_cholesky is None)
    is_diag = len(q_sqrt.shape) == 2

    M, L = tf.shape(q_mu)[0], tf.shape(q_mu)[1]

    if is_white:
        alpha = q_mu  # [M, L]
    else:
        if K is not None:
            Lp = tf.linalg.cholesky(K)  # [L, M, M] or [M, M]
        elif K_cholesky is not None:
            Lp = K_cholesky  # [L, M, M] or [M, M]

        is_batched = len(Lp.shape) == 3

        q_mu = tf.transpose(q_mu)[:, :, None] if is_batched else q_mu  # [L, M, 1] or [M, L]
        alpha = tf.linalg.triangular_solve(Lp, q_mu, lower=True)  # [L, M, 1] or [M, L]

    if is_diag:
        Lq = Lq_diag = q_sqrt
        Lq_full = tf.linalg.diag(tf.transpose(q_sqrt))  # [L, M, M]
    else:
        Lq = Lq_full = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [L, M, M]
        Lq_diag = tf.linalg.diag_part(Lq)  # [M, L]

    # Mahalanobis term: μqᵀ Σp⁻¹ μq
    mahalanobis = tf.reduce_sum(tf.square(alpha))

    # Constant term: - L * M
    constant = -to_default_float(tf.size(q_mu, out_type=tf.int64))

    # Log-determinant of the covariance of q(x):
    logdet_qcov = tf.reduce_sum(tf.math.log(tf.square(Lq_diag)))

    # Trace term: tr(Σp⁻¹ Σq)
    if is_white:
        trace = tf.reduce_sum(tf.square(Lq))
    else:
        if is_diag and not is_batched:
            # K is [M, M] and q_sqrt is [M, L]: fast specialisation
            LpT = tf.transpose(Lp)  # [M, M]
            Lp_inv = tf.linalg.triangular_solve(
                Lp, tf.eye(M, dtype=default_float()), lower=True
            )  # [M, M]
            K_inv = tf.linalg.diag_part(tf.linalg.triangular_solve(LpT, Lp_inv, lower=False))[
                :, None
            ]  # [M, M] -> [M, 1]
            trace = tf.reduce_sum(K_inv * tf.square(q_sqrt))
        else:
            if is_batched or Version(tf.__version__) >= Version("2.2"):
                Lp_full = Lp
            else:
                # workaround for segfaults when broadcasting in TensorFlow<2.2
                Lp_full = tf.tile(tf.expand_dims(Lp, 0), [L, 1, 1])
            LpiLq = tf.linalg.triangular_solve(Lp_full, Lq_full, lower=True)
            trace = tf.reduce_sum(tf.square(LpiLq))

    twoKL = mahalanobis + constant - logdet_qcov + trace

    # Log-determinant of the covariance of p(x):
    if not is_white:
        log_sqdiag_Lp = tf.math.log(tf.square(tf.linalg.diag_part(Lp)))
        sum_log_sqdiag_Lp = tf.reduce_sum(log_sqdiag_Lp)
        # If K is [L, M, M], num_latent_gps is no longer implicit, no need to multiply the single kernel logdet
        scale = 1.0 if is_batched else to_default_float(L)
        twoKL += scale * sum_log_sqdiag_Lp

    return 0.5 * twoKL
