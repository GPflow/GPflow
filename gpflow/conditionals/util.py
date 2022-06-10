# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

import tensorflow as tf

from ..base import MeanAndVariance
from ..config import default_float, default_jitter
from ..experimental.check_shapes import check_shape as cs
from ..experimental.check_shapes import check_shapes
from ..utilities.ops import leading_transpose


@check_shapes(
    "Kmn: [M, batch..., N]",
    "Kmm: [M, M]",
    "Knn: [batch..., N, N] if full_cov",
    "Knn: [batch..., N] if not full_cov",
    "f: [M, R]",
    "q_sqrt: [M_R_or_R_M_M...]",
    "return[0]: [batch..., N, R]",
    "return[1]: [batch..., R, N, N] if full_cov",
    "return[1]: [batch..., N, R] if not full_cov",
)
def base_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    r"""
    Given a g1 and g2, and distribution p and q such that::

      p(g2) = N(g2; 0, Kmm)

      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)

    And::

      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)

    This method computes the mean and (co)variance of::

      q(g1) = ∫ q(g2) p(g1 | g2)

    :param q_sqrt: If this is a Tensor, it must have shape [R, M, M] (lower
        triangular) or [M, R] (diagonal)
    :return: mean, variance
    """
    Lm = tf.linalg.cholesky(Kmm)
    return base_conditional_with_lm(
        Kmn=Kmn, Lm=Lm, Knn=Knn, f=f, full_cov=full_cov, q_sqrt=q_sqrt, white=white
    )


@check_shapes(
    "Kmn: [M, batch..., N]",
    "Lm: [M, M]",
    "Knn: [batch..., N, N] if full_cov",
    "Knn: [batch..., N] if not full_cov",
    "f: [M, R]",
    "q_sqrt: [M_R_or_R_M_M...]",
    "return[0]: [batch..., N, R]",
    "return[1]: [batch..., R, N, N] if full_cov",
    "return[1]: [batch..., N, R] if not full_cov",
)
def base_conditional_with_lm(
    Kmn: tf.Tensor,
    Lm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    r"""
    Has the same functionality as the `base_conditional` function, except that instead of
    `Kmm` this function accepts `Lm`, which is the Cholesky decomposition of `Kmm`.

    This allows `Lm` to be precomputed, which can improve performance.
    """
    if q_sqrt is not None:
        cs(q_sqrt, "[M, R]" if q_sqrt.shape.ndims == 2 else "[R, M, M]")

    # compute kernel stuff
    num_func = tf.shape(f)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(f)[-2]

    # get the leading dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1]),
        ],
        0,
    )  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]

    leading_dims = tf.shape(Kmn)[:-2]

    # Compute the projection matrix A
    Lm = tf.broadcast_to(Lm, tf.concat([leading_dims, tf.shape(Lm)], 0))  # [..., M, M]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [..., M, N]

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.linalg.matmul(A, A, transpose_a=True)  # [..., N, N]
        cov_shape = tf.concat([leading_dims, [num_func, N, N]], 0)
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -3), cov_shape)  # [..., R, N, N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), -2)  # [..., N]
        cov_shape = tf.concat([leading_dims, [num_func, N]], 0)  # [..., R, N]
        fvar = tf.broadcast_to(tf.expand_dims(fvar, -2), cov_shape)  # [..., R, N]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)

    # construct the conditional mean
    f_shape = tf.concat([leading_dims, [M, num_func]], 0)  # [..., M, R]
    f = tf.broadcast_to(f, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L = tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
            L_shape = tf.shape(L)
            L = tf.broadcast_to(L, tf.concat([leading_dims, L_shape], 0))

            shape = tf.concat([leading_dims, [num_func, M, N]], axis=0)
            A_tiled = tf.broadcast_to(tf.expand_dims(A, -3), shape)
            LTA = tf.linalg.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" % str(q_sqrt.shape.ndims))

        if full_cov:
            fvar = fvar + tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), -2)  # [R, N]

    if not full_cov:
        fvar = tf.linalg.adjoint(fvar)  # [N, R]

    return fmean, fvar


@check_shapes(
    "mean: [batch..., N, D]",
    "cov: [batch..., N, D, D] if full_cov",
    "cov: [batch..., N, D] if not full_cov",
    "return: [batch..., N, D] if num_samples is None",
    "return: [batch..., S, N, D] if num_samples is not None",
)
def sample_mvn(
    mean: tf.Tensor, cov: tf.Tensor, full_cov: bool, num_samples: Optional[int] = None
) -> tf.Tensor:
    """
    Returns a sample from a D-dimensional Multivariate Normal distribution.

    :return: sample from the MVN
    """
    mean_shape = tf.shape(mean)
    S = num_samples if num_samples is not None else 1
    D = mean_shape[-1]
    leading_dims = mean_shape[:-2]

    if not full_cov:
        # mean: [..., N, D] and cov [..., N, D]
        eps_shape = tf.concat([leading_dims, [S], mean_shape[-2:]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., S, N, D]
        samples = mean[..., None, :, :] + tf.sqrt(cov)[..., None, :, :] * eps  # [..., S, N, D]

    else:
        # mean: [..., N, D] and cov [..., N, D, D]
        jittermat = (
            tf.eye(D, batch_shape=mean_shape[:-1], dtype=default_float()) * default_jitter()
        )  # [..., N, D, D]
        eps_shape = tf.concat([mean_shape, [S]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., N, D, S]
        chol = tf.linalg.cholesky(cov + jittermat)  # [..., N, D, D]
        samples = mean[..., None] + tf.linalg.matmul(chol, eps)  # [..., N, D, S]
        samples = leading_transpose(samples, [..., -1, -3, -2])  # [..., S, N, D]

    if num_samples is None:
        return tf.squeeze(samples, axis=-3)  # [..., N, D]
    return samples  # [..., S, N, D]


@check_shapes(
    "fvar: [batch..., P, N, N] if full_cov",
    "fvar: [batch..., N, P] if not full_cov",
    "return: [batch..., N, P, N, P] if full_cov and full_output_cov",
    "return: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
)
def expand_independent_outputs(fvar: tf.Tensor, full_cov: bool, full_output_cov: bool) -> tf.Tensor:
    """
    Reshapes fvar to the correct shape, specified by `full_cov` and `full_output_cov`.

    :param fvar: Single-output covariance.
    :return: Multi-output covariance.
    """
    if full_cov and full_output_cov:
        fvar = tf.linalg.diag(tf.transpose(fvar))  # [N, N, P, P]
        fvar = tf.transpose(fvar, [0, 2, 1, 3])  # [N, P, N, P]
    if not full_cov and full_output_cov:
        fvar = tf.linalg.diag(fvar)  # [N, P, P]
    if full_cov and not full_output_cov:
        pass  # [P, N, N]
    if not full_cov and not full_output_cov:
        pass  # [N, P]

    return fvar


@check_shapes(
    "Kmn: [M, L, N, P]",
    "Kmm: [L, M, M]",
    "Knn: [N, P] if (not full_cov) and (not full_output_cov)",
    "Knn: [P, N, N] if full_cov and (not full_output_cov)",
    "Knn: [N, P, P] if (not full_cov) and full_output_cov",
    "Knn: [N, P, N, P] if full_cov and full_output_cov",
    "f: [M, L]",
    "q_sqrt: [M_L_or_L_M_M...]",
    "return[0]: [N, P]",
    "return[1]: [N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [N, P, N, P] if full_cov and full_output_cov",
)
def independent_interdomain_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    The inducing outputs live in the g-space (R^L).

    Interdomain conditional calculation.

    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return: mean, variance
    """
    M, L, N, P = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)

    Lm = tf.linalg.cholesky(Kmm)  # [L, M, M]

    # Compute the projection matrix A
    Kmn = tf.reshape(tf.transpose(Kmn, (1, 0, 2, 3)), (L, M, N * P))
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [L, M, M] \ [L, M, N*P] -> [L, M, N*P]
    Ar = tf.reshape(A, (L, M, N, P))

    # compute the covariance due to the conditioning
    if full_cov and full_output_cov:
        fvar = Knn - tf.tensordot(Ar, Ar, [[0, 1], [0, 1]])  # [N, P, N, P]
    elif full_cov and not full_output_cov:
        At = tf.reshape(tf.transpose(Ar), (P, N, M * L))  # [P, N, L]
        fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [P, N, N]
    elif not full_cov and full_output_cov:
        At = tf.reshape(tf.transpose(Ar, [2, 3, 1, 0]), (N, P, M * L))  # [N, P, L]
        fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [N, P, P]
    elif not full_cov and not full_output_cov:
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, P))  # Knn: [N, P]

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(
            Lm, A, adjoint=True
        )  # [L, M, M] \ [L, M, N*P]  ->  [L, M, N*P]
        Ar = tf.reshape(A, (L, M, N, P))

    fmean = tf.tensordot(Ar, f, [[1, 0], [0, 1]])  # [N, P]

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 3:
            Lf = tf.linalg.band_part(q_sqrt, -1, 0)  # [L, M, M]
            LTA = tf.linalg.matmul(
                Lf, A, transpose_a=True
            )  # [L, M, M]  *  [L, M, P]  ->  [L, M, P]
        else:  # q_sqrt [M, L]
            LTA = A * tf.transpose(q_sqrt)[..., None]  # [L, M, P]

        if full_cov and full_output_cov:
            LTAr = tf.reshape(LTA, (L * M, N * P))
            fvar = fvar + tf.reshape(tf.linalg.matmul(LTAr, LTAr, transpose_a=True), (N, P, N, P))
        elif full_cov and not full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [2, 0, 1])  # [P, M, N]
            fvar = fvar + tf.linalg.matmul(LTAr, LTAr, transpose_a=True)  # [P, N, N]
        elif not full_cov and full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [1, 0, 2])  # [N, M, P]
            fvar = fvar + tf.linalg.matmul(LTAr, LTAr, transpose_a=True)  # [N, P, P]
        elif not full_cov and not full_output_cov:
            fvar = fvar + tf.reshape(tf.reduce_sum(tf.square(LTA), (0, 1)), (N, P))

    return fmean, fvar


@check_shapes(
    "Kmn: [M, N, P]",
    "Kmm: [M, M]",
    "Knn: [N, P] if (not full_cov) and (not full_output_cov)",
    "Knn: [P, N, N] if full_cov and (not full_output_cov)",
    "Knn: [N, P, P] if (not full_cov) and full_output_cov",
    "Knn: [N, P, N, P] if full_cov and full_output_cov",
    "f: [M, 1]",
    "q_sqrt: [_1_L_or_1_M_M...]",
    "return[0]: [N, P]",
    "return[1]: [N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [N, P, N, P] if full_cov and full_output_cov",
)
def fully_correlated_conditional(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.

    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return: mean, variance
    """
    mean, var = fully_correlated_conditional_repeat(
        Kmn,
        Kmm,
        Knn,
        f,
        full_cov=full_cov,
        full_output_cov=full_output_cov,
        q_sqrt=q_sqrt,
        white=white,
    )
    return tf.squeeze(mean, axis=0), tf.squeeze(var, axis=0)


@check_shapes(
    "Kmn: [M, N, P]",
    "Kmm: [M, M]",
    "Knn: [N, P] if (not full_cov) and (not full_output_cov)",
    "Knn: [P, N, N] if full_cov and (not full_output_cov)",
    "Knn: [N, P, P] if (not full_cov) and full_output_cov",
    "Knn: [N, P, N, P] if full_cov and full_output_cov",
    "f: [M, R]",
    "q_sqrt: [M_R_or_R_M_M...]",
    "return[0]: [R, N, P]",
    "return[1]: [R, N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [R, P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [R, N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [R, N, P, N, P] if full_cov and full_output_cov",
)
def fully_correlated_conditional_repeat(
    Kmn: tf.Tensor,
    Kmm: tf.Tensor,
    Knn: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    full_output_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.
    Note: This conditional can handle 'repetitions' R, given in `f` and `q_sqrt`.

    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return: mean, variance
    """
    R = tf.shape(f)[1]
    M, N, P = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)

    Lm = tf.linalg.cholesky(Kmm)

    # Compute the projection matrix A
    # Lm: [M, M]    Kmn: [M, P]
    Kmn = tf.reshape(Kmn, (M, N * P))  # [M, P]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [M, P]
    Ar = tf.reshape(A, (M, N, P))

    # compute the covariance due to the conditioning
    if full_cov and full_output_cov:
        # fvar = Knn - tf.linalg.matmul(Ar, Ar, transpose_a=True)  # [P, P], then reshape?
        fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # [N, P, N, P]
    elif full_cov and not full_output_cov:
        At = tf.transpose(Ar)  # [P, N, M]
        fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [P, N, N]
    elif not full_cov and full_output_cov:
        # This transpose is annoying
        At = tf.transpose(Ar, [1, 0, 2])  # [N, M, P]
        # fvar = Knn - tf.einsum('mnk,mnl->nkl', Ar, Ar)
        fvar = Knn - tf.linalg.matmul(At, At, transpose_a=True)  # [N, P, P]
    elif not full_cov and not full_output_cov:
        # Knn: [N, P]
        # Can also do this with a matmul
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0]), (N, P))

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.linalg.triangular_solve(Lm, A, adjoint=True)  # [M, P]

    # f: [M, R]
    fmean = tf.linalg.matmul(f, A, transpose_a=True)  # [R, M]  *  [M, P]  ->  [R, P]
    fmean = tf.reshape(fmean, (R, N, P))  # [R, N, P]

    if q_sqrt is not None:
        Lf = tf.linalg.band_part(q_sqrt, -1, 0)  # [R, M, M]
        if q_sqrt.shape.ndims == 3:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # [R, M, P]
            LTA = tf.linalg.matmul(Lf, A_tiled, transpose_a=True)  # [R, M, P]
        elif q_sqrt.shape.ndims == 2:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # [R, M, P]
            LTA = Lf * A_tiled  # [R, M, P]
        else:  # pragma: no cover
            raise ValueError(f"Bad dimension for q_sqrt: {q_sqrt.shape.ndims}")

        if full_cov and full_output_cov:
            addvar = tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, P, P]
            fvar = fvar[None, :, :, :, :] + tf.reshape(addvar, (R, N, P, N, P))
        elif full_cov and not full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, [R, M, N, P]), [0, 3, 1, 2])  # [R, P, M, N]
            addvar = tf.linalg.matmul(LTAr, LTAr, transpose_a=True)  # [R, P, N, N]
            fvar = fvar[None, ...] + addvar  # [R, P, N, N]
        elif not full_cov and full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (R, M, N, P)), [0, 2, 3, 1])  # [R, N, P, M]
            fvar = fvar[None, ...] + tf.linalg.matmul(LTAr, LTAr, transpose_b=True)  # [R, N, P, P]
        elif not full_cov and not full_output_cov:
            addvar = tf.reshape(tf.reduce_sum(tf.square(LTA), axis=1), (R, N, P))  # [R, N, P]
            fvar = fvar[None, ...] + addvar  # [R, N, P]
    else:
        fvar_shape = tf.concat([[R], tf.shape(fvar)], axis=0)
        fvar = tf.broadcast_to(fvar[None], fvar_shape)

    return fmean, fvar


@check_shapes(
    "A: [left..., right...]",
    "return: [right..., left...]",
)
def rollaxis_left(A: tf.Tensor, num_rolls: int) -> tf.Tensor:
    """Roll the tensor `A` backwards `num_rolls` times."""
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([num_rolls + tf.range(rank - num_rolls), tf.range(num_rolls)], 0)
    return tf.transpose(A, perm)


@check_shapes(
    "A: [left..., right...]",
    "return: [right..., left...]",
)
def rollaxis_right(A: tf.Tensor, num_rolls: int) -> tf.Tensor:
    """Roll the tensor `A` forward `num_rolls` times."""
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([rank - num_rolls + tf.range(num_rolls), tf.range(rank - num_rolls)], 0)
    return tf.transpose(A, perm)


@check_shapes(
    "W: [P, L]",
    "g_mean: [batch..., N, L]",
    "g_var: [batch..., N, L] if not full_cov",
    "g_var: [L, batch..., N, N] if full_cov",
    "return[0]: [batch..., N, P]",
    "return[1]: [batch..., N, P] if (not full_cov) and (not full_output_cov)",
    "return[1]: [batch..., P, N, N] if full_cov and (not full_output_cov)",
    "return[1]: [batch..., N, P, P] if (not full_cov) and full_output_cov",
    "return[1]: [batch..., N, P, N, P] if full_cov and full_output_cov",
)
def mix_latent_gp(
    W: tf.Tensor, g_mean: tf.Tensor, g_var: tf.Tensor, full_cov: bool, full_output_cov: bool
) -> MeanAndVariance:
    r"""Takes the mean and variance of an uncorrelated L-dimensional latent GP
    and returns the mean and the variance of the mixed GP, `f = W g`,
    where both f and g are GPs.

    :return: f_mean and f_var
    """
    f_mean = tf.tensordot(g_mean, W, [[-1], [-1]])  # [..., N, P]

    if full_cov and full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        g_var = rollaxis_left(g_var, 1)  # [..., N, N, L]

        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, N, P, P]
        f_var = leading_transpose(f_var, [..., -4, -2, -3, -1])  # [..., N, P, N, P]

    elif full_cov and not full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        f_var = tf.tensordot(g_var, W ** 2, [[0], [-1]])  # [..., N, N, P]
        f_var = leading_transpose(f_var, [..., -1, -3, -2])  # [..., P, N, N]

    elif not full_cov and full_output_cov:  # g_var is [..., N, L]
        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, P, P]

    elif not full_cov and not full_output_cov:  # g_var is [..., N, L]
        W_squared = W ** 2  # [P, L]
        f_var = tf.tensordot(g_var, W_squared, [[-1], [-1]])  # [..., N, P]

    return f_mean, f_var


@check_shapes(
    "Kmns: [P, M, batch..., N]",
    "Kmms: [P, M, M]",
    "Knns: [P, batch..., N, N] if full_cov",
    "Knns: [P, batch..., N] if not full_cov",
    "f: [M, P]",
    "q_sqrt: [M_R_or_R_M_M...]",
    "return[0]: [batch..., N, R]",
    "return[1]: [batch..., R, N, N] if full_cov",
    "return[1]: [batch..., N, R] if not full_cov",
)
def separate_independent_conditional_implementation(
    Kmns: tf.Tensor,
    Kmms: tf.Tensor,
    Knns: tf.Tensor,
    f: tf.Tensor,
    *,
    full_cov: bool = False,
    q_sqrt: Optional[tf.Tensor] = None,
    white: bool = False,
) -> MeanAndVariance:
    """
    Multi-output GP with independent GP priors.

    Number of latent processes equals the number of outputs (L = P).

    Further reference:

    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multioutput notebook for more information about the multioutput framework.
    - See above for the parameters and the return value.
    """
    fs = tf.transpose(f)[:, :, None]  # [P, M, 1]
    # [P, 1, M, M]  or  [P, M, 1]

    if q_sqrt is not None:
        q_sqrts = (
            tf.transpose(q_sqrt)[:, :, None] if q_sqrt.shape.ndims == 2 else q_sqrt[:, None, :, :]
        )
        base_conditional_args_to_map = (
            Kmms,
            Kmns,
            Knns,
            fs,
            q_sqrts,
        )  # type: Tuple[tf.Tensor, ...]

        def single_gp_conditional(
            t: Tuple[tf.Tensor, ...]
        ) -> MeanAndVariance:  # pragma: no cover - tf.map_fn is invisible to codecov
            Kmm, Kmn, Knn, f, q_sqrt = t
            return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    else:
        base_conditional_args_to_map = (Kmms, Kmns, Knns, fs)

        def single_gp_conditional(
            t: Tuple[tf.Tensor, ...]
        ) -> MeanAndVariance:  # pragma: no cover - tf.map_fn is invisible to codecov
            Kmm, Kmn, Knn, f = t
            return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    rmu, rvar = tf.map_fn(
        single_gp_conditional, base_conditional_args_to_map, (default_float(), default_float())
    )  # [P, N, 1], [P, 1, N, N] or [P, N, 1]

    fmu = rollaxis_left(tf.squeeze(rmu, axis=-1), 1)  # [N, P]

    if full_cov:
        fvar = tf.squeeze(rvar, axis=-3)  # [..., 0, :, :]  # [P, N, N]
    else:
        fvar = rollaxis_left(tf.squeeze(rvar, axis=-1), 1)  # [N, P]

    return fmu, fvar
