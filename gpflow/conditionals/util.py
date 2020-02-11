import tensorflow as tf

from ..config import default_float, default_jitter
from ..utilities.ops import leading_transpose


def base_conditional(Kmn: tf.Tensor,
                     Kmm: tf.Tensor,
                     Knn: tf.Tensor,
                     function: tf.Tensor,
                     *,
                     full_cov=False,
                     q_sqrt=None,
                     white=False):
    r"""
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2; 0, Kmm)

      p(g1) = N(g1; 0, Knn)
      p(g1 | g2) = N(g1; Knm (Kmm⁻¹) g2, Knn - Knm (Kmm⁻¹) Kmn)

    And
      q(g2) = N(g2; f, q_sqrt q_sqrtᵀ)

    This method computes the mean and (co)variance of
      q(g1) = ∫ q(g2) p(g1 | g2)

    :param Kmn: [M, ..., N]
    :param Kmm: [M, M]
    :param Knn: [..., N, N]  or  N
    :param f: [M, R]
    :param full_cov: bool
    :param q_sqrt: None or [R, M, M] (lower triangular)
    :param white: bool
    :return: [N, R]  or [R, N, N]
    """
    # compute kernel stuff
    num_func = tf.shape(function)[-1]  # R
    N = tf.shape(Kmn)[-1]
    M = tf.shape(function)[-2]

    # get the leadings dims in Kmn to the front of the tensor
    # if Kmn has rank two, i.e. [M, N], this is the identity op.
    K = tf.rank(Kmn)
    perm = tf.concat(
        [
            tf.reshape(tf.range(1, K - 1), [K - 2]),  # leading dims (...)
            tf.reshape(0, [1]),  # [M]
            tf.reshape(K - 1, [1])
        ],
        0)  # [N]
    Kmn = tf.transpose(Kmn, perm)  # [..., M, N]

    leading_dims = tf.shape(Kmn)[:-2]
    Lm = tf.linalg.cholesky(Kmm)  # [M, M]

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
    f = tf.broadcast_to(function, f_shape)  # [..., M, R]
    fmean = tf.linalg.matmul(A, f, transpose_a=True)  # [..., N, R]

    if q_sqrt is not None:
        q_sqrt_dims = q_sqrt.shape.ndims
        if q_sqrt_dims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt_dims == 3:
            L =  tf.linalg.band_part(q_sqrt, -1, 0)  # force lower triangle # [R, M, M]
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

    return fmean, fvar  # [N, R], [R, N, N] or [N, R]


def sample_mvn(mean, cov, cov_structure=None, num_samples=None):
    """
    Returns a sample from a D-dimensional Multivariate Normal distribution
    :param mean: [..., N, D]
    :param cov: [..., N, D] or [..., N, D, D]
    :param cov_structure: "diag" or "full"
    - "diag": cov holds the diagonal elements of the covariance matrix
    - "full": cov holds the full covariance matrix (without jitter)
    :return: sample from the MVN of shape [..., (S), N, D], S = num_samples
    """
    assert cov_structure == "diag" or cov_structure == "full"

    mean_shape = tf.shape(mean)
    S = num_samples if num_samples is not None else 1
    D = mean_shape[-1]
    leading_dims = mean_shape[:-2]

    if cov_structure == "diag":
        # mean: [..., N, D] and cov [..., N, D]
        tf.assert_equal(tf.rank(mean), tf.rank(cov))
        eps_shape = tf.concat([leading_dims, [S], mean_shape[-2:]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., S, N, D]
        samples = mean[..., None, :, :] + tf.sqrt(cov)[..., None, :, :] * eps  # [..., S, N, D]
    elif cov_structure == "full":
        # mean: [..., N, D] and cov [..., N, D, D]
        tf.assert_equal(tf.rank(mean) + 1, tf.rank(cov))
        jittermat = (tf.eye(D, batch_shape=mean_shape[:-1], dtype=default_float()) * default_jitter()
                     )  # [..., N, D, D]
        eps_shape = tf.concat([mean_shape, [S]], 0)
        eps = tf.random.normal(eps_shape, dtype=default_float())  # [..., N, D, S]
        chol = tf.linalg.cholesky(cov + jittermat)  # [..., N, D, D]
        samples = mean[..., None] + tf.linalg.matmul(chol, eps)  # [..., N, D, S]
        samples = leading_transpose(samples, [..., -1, -3, -2])  # [..., S, N, D]

    if num_samples is None:
        return samples[..., 0, :, :]  # [..., N, D]
    return samples  # [..., S, N, D]


def expand_independent_outputs(fvar, full_cov, full_output_cov):
    """
    Reshapes fvar to the correct shape, specified by `full_cov` and `full_output_cov`.

    :param fvar: has shape [N, P] (full_cov = False) or [P, N, N] (full_cov = True).
    :return:
    1. full_cov: True and full_output_cov: True
       fvar [N, P, N, P]
    2. full_cov: True and full_output_cov: False
       fvar [P, N, N]
    3. full_cov: False and full_output_cov: True
       fvar [N, P, P]
    4. full_cov: False and full_output_cov: False
       fvar [N, P]
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


def independent_interdomain_conditional(Kmn,
                                        Kmm,
                                        Knn,
                                        f,
                                        *,
                                        full_cov=False,
                                        full_output_cov=False,
                                        q_sqrt=None,
                                        white=False):
    """
    The inducing outputs live in the g-space (R^L).
    Interdomain conditional calculation.
    :param Kmn: [M, L, N, P]
    :param Kmm: [L, M, M]
    :param Knn: [N, P]  or  [N, N]  or  [P, N, N]  or  [N, P, N, P]
    :param f: data matrix, [M, L]
    :param q_sqrt: [L, M, M]  or  [M, L]
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: [N, P]
        - variance: [N, P], [N, P, P], [P, N, N], [N, P, N, P]
    """
    M, L, N, P = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)

    Lm = tf.linalg.cholesky(Kmm)  # [L, M, M]

    # Compute the projection matrix A
    Kmn = tf.reshape(tf.transpose(Kmn, (1, 0, 2, 3)), (L, M, N * P))
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [L, M, M]  *  [L, M, P]  ->  [L, M, P]
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
        A = tf.linalg.triangular_solve(Lm, Ar)  # [L, M, M]  *  [L, M, P]  ->  [L, M, P]
        Ar = tf.reshape(A, (L, M, N, P))

    fmean = tf.tensordot(Ar, f, [[1, 0], [0, 1]])  # [N, P]

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 3:
            Lf = tf.linalg.band_part(q_sqrt, -1, 0)  # [L, M, M]
            LTA = tf.linalg.matmul(Lf, A, transpose_a=True)  # [L, M, M]  *  [L, M, P]  ->  [L, M, P]
        else:  # q_sqrt [M, L]
            LTA = (A * tf.transpose(q_sqrt)[..., None])  # [L, M, P]

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


def fully_correlated_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.
    :param Kmn: [M, N, P]
    :param Kmm: [M, M]
    :param Knn: [N, P] or [N, P, N, P]
    :param f: data matrix, [M, 1]
    :param q_sqrt: [1, M, M]  or [1, L]
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: [N, P]
        - variance: [N, P], [N, P, P], [P, N, N], [N, P, N, P]
    """
    m, v = fully_correlated_conditional_repeat(Kmn,
                                               Kmm,
                                               Knn,
                                               f,
                                               full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt,
                                               white=white)
    return m[0, ...], v[0, ...]


def fully_correlated_conditional_repeat(Kmn,
                                        Kmm,
                                        Knn,
                                        f,
                                        *,
                                        full_cov=False,
                                        full_output_cov=False,
                                        q_sqrt=None,
                                        white=False):
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.
    Note: This conditional can handle 'repetitions' R, given in `f` and `q_sqrt`.
    :param Kmn: [M, N, P]
    :param Kmm: [M, M]
    :param Knn: [N, P] or [N, P, N, P]
    :param f: data matrix, [M, R]
    :param q_sqrt: [R, M, M]  or [R, L]
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: [R, N, P]
        - variance: [R, N, P], [R, N, P, P], [R, P, N, N], [R, N, P, N, P]
    """
    R = tf.shape(f)[1]
    M, N, K = tf.unstack(tf.shape(Kmn), num=Kmn.shape.ndims, axis=0)
    Lm = tf.linalg.cholesky(Kmm)

    # Compute the projection matrix A
    # Lm: [M, M]    Kmn: [M, K]
    Kmn = tf.reshape(Kmn, (M, N * K))  # [M, K]
    A = tf.linalg.triangular_solve(Lm, Kmn, lower=True)  # [M, K]
    Ar = tf.reshape(A, (M, N, K))

    # compute the covariance due to the conditioning
    if full_cov and full_output_cov:
        # fvar = Knn - tf.linalg.matmul(Ar, Ar, transpose_a=True)  # [K, K], then reshape?
        fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # [N, K, N, K]
    elif full_cov and not full_output_cov:
        At = tf.transpose(Ar)  # [K, N, M]
        fvar = Knn - tf.linalg.matmul(At, At, transpose_b=True)  # [K, N, N]
    elif not full_cov and full_output_cov:
        # This transpose is annoying
        At = tf.transpose(Ar, [1, 0, 2])  # [N, M, K]
        # fvar = Knn - tf.einsum('mnk,mnl->nkl', Ar, Ar)
        fvar = Knn - tf.linalg.matmul(At, At, transpose_a=True)  # [N, K, K]
    elif not full_cov and not full_output_cov:
        # Knn: [N, K]
        # Can also do this with a matmul
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0]), (N, K))

    # another backsubstitution in the unwhitened case
    if not white:
        # A = tf.linalg.triangular_solve(tf.linalg.adjoint(Lm), A, lower=False)  # [M, K]
        raise NotImplementedError("Need to verify this.")  # pragma: no cover

    # f: [M, R]
    fmean = tf.linalg.matmul(f, A, transpose_a=True)  # [R, M]  *  [M, K]  ->  [R, K]
    fmean = tf.reshape(fmean, (R, N, K))  # [R, N, K]

    if q_sqrt is not None:
        Lf = tf.linalg.band_part(q_sqrt, -1, 0)  # [R, M, M]
        if q_sqrt.shape.ndims == 3:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # [R, M, K]
            LTA = tf.linalg.matmul(Lf, A_tiled, transpose_a=True)  # [R, M, K]
        elif q_sqrt.shape.ndims == 2:  # pragma: no cover
            raise NotImplementedError("Does not support diagonal q_sqrt yet...")
        else:  # pragma: no cover
            raise ValueError(f"Bad dimension for q_sqrt: {q_sqrt.shape.ndims}")

        if full_cov and full_output_cov:
            addvar = tf.linalg.matmul(LTA, LTA, transpose_a=True)  # [R, K, K]
            fvar = fvar[None, :, :, :, :] + tf.reshape(addvar, (R, N, K, N, K))
        elif full_cov and not full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, [R, M, N, K]), [0, 3, 1, 2])  # [R, K, M, N]
            addvar = tf.linalg.matmul(LTAr, LTAr, transpose_a=True)  # [R, K, N, N]
            fvar = fvar[None, ...] + addvar  # [R, K, N, N]
        elif not full_cov and full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (R, M, N, K)), [0, 2, 3, 1])  # [R, N, K, M]
            fvar = fvar[None, ...] + tf.linalg.matmul(LTAr, LTAr, transpose_b=True)  # [R, N, K, K]
        elif not full_cov and not full_output_cov:
            addvar = tf.reshape(tf.reduce_sum(tf.square(LTA), axis=1), (R, N, K))  # [R, N, K]
            fvar = fvar[None, ...] + addvar  # [R, N, K]
    else:
        fvar = tf.broadcast_to(fvar[None], tf.shape(fmean))
    return fmean, fvar


def rollaxis_left(A, num_rolls):
    """Roll the tensor `A` backwards `num_rolls` times."""
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([num_rolls + tf.range(rank - num_rolls), tf.range(num_rolls)], 0)
    return tf.transpose(A, perm)


def rollaxis_right(A, num_rolls):
    """Roll the tensor `A` forward `num_rolls` times."""
    assert num_rolls > 0
    rank = tf.rank(A)
    perm = tf.concat([rank - num_rolls + tf.range(num_rolls), tf.range(rank - num_rolls)], 0)
    return tf.transpose(A, perm)


def mix_latent_gp(W, g_mu, g_var, full_cov, full_output_cov):
    r"""Takes the mean and variance of an uncorrelated L-dimensional latent GP
    and returns the mean and the variance of the mixed GP, `f = W g`,
    where both f and g are GPs, with W having a shape [P, L]

    :param W: [P, L]
    :param g_mu: [..., N, L]
    :param g_var: [..., N, L] (full_cov = False) or [L, ..., N, N] (full_cov = True)
    :return: f_mu and f_var, shape depends on `full_cov` and `full_output_cov`
    """
    f_mu = tf.tensordot(g_mu, W, [[-1], [-1]])  # [..., N, P]

    if full_cov and full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        g_var = rollaxis_left(g_var, 1)  # [..., N, N, L]
        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, N, P, P]
        f_var = leading_transpose(f_var, [..., -4, -2, -3, -1])  # [..., N, P, N, P]

    elif full_cov and not full_output_cov:  # g_var is [L, ..., N, N]
        # this branch is practically never taken
        f_var = tf.tensordot(g_var, W**2, [[0], [-1]])  # [..., N, N, P]
        f_var = leading_transpose(f_var, [..., -1, -3, -2])  # [..., P, N, N]

    elif not full_cov and full_output_cov:  # g_var is [..., N, L]
        g_var = tf.expand_dims(g_var, axis=-2)  # [..., N, 1, L]
        g_var_W = g_var * W  # [..., N, P, L]
        f_var = tf.tensordot(g_var_W, W, [[-1], [-1]])  # [..., N, P, P]

    elif not full_cov and not full_output_cov:  # g_var is [..., N, L]
        W_squared = W**2  # [P, L]
        f_var = tf.tensordot(g_var, W_squared, [[-1], [-1]])  # [..., N, P]

    return f_mu, f_var
