from typing import Callable

import tensorflow as tf

from ..util import create_logger, default_jitter

logger = create_logger()


def base_conditional(
        Kmn: tf.Tensor,
        Kmm: tf.Tensor,
        Knn: tf.Tensor,
        function: tf.Tensor,
        *, full_cov=False, q_sqrt=None, white=False):
    """
    Given a g1 and g2, and distribution p and q such that
      p(g2) = N(g2;0,Kmm)
      p(g1) = N(g1;0,Knn)
      p(g1|g2) = N(g1;0,Knm)
    And
      q(g2) = N(g2;f,q_sqrt*q_sqrt^T)
    This method computes the mean and (co)variance of
      q(g1) = \int q(g2) p(g1|g2)
    :param Kmn: M x N
    :param Kmm: M x M
    :param Knn: N x N  or  N
    :param f: M x R
    :param full_cov: bool
    :param q_sqrt: None or R x M x M (lower triangular)
    :param white: bool
    :return: N x R  or R x N x N
    """
    logger.debug("base conditional")

    # compute kernel stuff
    num_func = tf.shape(function)[1]  # [R]
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix [A]
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # Compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        fvar = tf.tile(fvar[None, :, :], [num_func, 1, 1])  # [R, N, N]
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        fvar = tf.tile(fvar[None, :], [num_func, 1])  # [R, N]

    # Another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # Construct the conditional mean
    fmean = tf.matmul(A, function, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # [R, M, N]
        elif q_sqrt.shape.ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # [R, M, M]
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # [R, M, N]
        else:  # pragma: no cover
            raise ValueError(f"Bad dimension for q_sqrt: {q_sqrt.shape.ndims}")
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # [R, N, N]
        else:
            fvar = fvar + tf.reduce_sum(LTA ** 2, 1)  # [R, N]

    if not full_cov:
        fvar = tf.transpose(fvar)  # [N, R]

    return fmean, fvar  # [N, R], [R, N, N] or [N, R]


def sample_mvn(mean, cov, cov_structure):
    """
    Returns a sample from a D-dimensional Multivariate Normal distribution
    :param mean: N x D
    :param cov: N x D or N x D x D
    :param cov_structure: "diag" or "full"
    - "diag": cov holds the diagonal elements of the covariance matrix
    - "full": cov holds the full covariance matrix (without jitter)
    :return: sample from the MVN of shape N x D
    """
    eps = tf.random_normal(mean.shape, dtype=mean.dtype)  # N x P
    if cov_structure == "diag":
        sample = mean + tf.sqrt(cov) * eps  # N x P
    elif cov_structure == "full":
        cov = cov + (tf.eye(mean.shape[1], dtype=mean.dtype) * default_jitter())[None, ...]  # N x P x P
        chol = tf.cholesky(cov)  # N x P x P
        return mean + (tf.matmul(chol, eps[..., None])[..., 0])  # N x P
    else:
        raise NotImplementedError

    return sample  # N x P


def expand_independent_outputs(fvar, full_cov, full_output_cov):
    """
    Reshapes fvar to the correct shape, specified by `full_cov` and `full_output_cov`.

    :param fvar: has shape N x P (full_cov = False) or P x N x N (full_cov = True).
    :return:
    1. full_cov: True and full_output_cov: True
       fvar N x P x N x P
    2. full_cov: True and full_output_cov: False
       fvar P x N x N
    3. full_cov: False and full_output_cov: True
       fvar N x P x P
    4. full_cov: False and full_output_cov: False
       fvar N x P
    """
    if full_cov and full_output_cov:
        fvar = tf.matrix_diag(tf.transpose(fvar))   # N x N x P x P
        fvar = tf.transpose(fvar, [0, 2, 1, 3])  # N x P x N x P
    if not full_cov and full_output_cov:
        fvar = tf.matrix_diag(fvar)   # N x P x P
    if full_cov and not full_output_cov:
        pass  # P x N x N
    if not full_cov and not full_output_cov:
        pass  # N x P

    return fvar


def independent_interdomain_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_output_cov=False,
                                        q_sqrt=None, white=False):
    """
    The inducing outputs live in the g-space (R^L).
    Interdomain conditional calculation.

    :param Kmn: M x L x N x P
    :param Kmm: L x M x M
    :param Knn: N x P  or  N x N  or  P x N x N  or  N x P x N x P
    :param f: data matrix, M x L
    :param q_sqrt: L x M x M  or  M x L
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: N x P
        - variance: N x P, N x P x P, P x N x N, N x P x N x P
    """
    logger.debug("independent_interdomain_conditional")
    M, L, N, P = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]

    Lm = tf.cholesky(Kmm)  # L x M x M

    # Compute the projection matrix A
    Kmn = tf.reshape(tf.transpose(Kmn, (1, 0, 2, 3)), (L, M, N * P))
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # L x M x M  *  L x M x NP  ->  L x M x NP
    Ar = tf.reshape(A, (L, M, N, P))

    # compute the covariance due to the conditioning
    if full_cov and full_output_cov:
        fvar = Knn - tf.tensordot(Ar, Ar, [[0, 1], [0, 1]])  # N x P x N x P
    elif full_cov and not full_output_cov:
        At = tf.reshape(tf.transpose(Ar), (P, N, M * L))  # P x N x ML
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # P x N x N
    elif not full_cov and full_output_cov:
        At = tf.reshape(tf.transpose(Ar, [2, 3, 1, 0]), (N, P, M * L))  # N x P x ML
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # N x P x P
    elif not full_cov and not full_output_cov:
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, P))  # Knn: N x P

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(Lm, Ar)  # L x M x M  *  L x M x NP  ->  L x M x NP
        Ar = tf.reshape(A, (L, M, N, P))

    fmean = tf.tensordot(Ar, f, [[1, 0], [0, 1]])  # N x P

    if q_sqrt is not None:
        if q_sqrt.shape.ndims == 3:
            Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # L x M x M
            LTA = tf.matmul(Lf, A, transpose_a=True)  # L x M x M  *  L x M x NP  ->  L x M x NP
        else:  # q_sqrt M x L
            LTA = (A * tf.transpose(q_sqrt)[..., None])  # L x M x NP

        if full_cov and full_output_cov:
            LTAr = tf.reshape(LTA, (L * M, N * P))
            fvar = fvar + tf.reshape(tf.matmul(LTAr, LTAr, transpose_a=True), (N, P, N, P))
        elif full_cov and not full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [2, 0, 1])  # P x LM x N
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # P x N x N
        elif not full_cov and full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [1, 0, 2])  # N x LM x P
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # N x P x P
        elif not full_cov and not full_output_cov:
            fvar = fvar + tf.reshape(tf.reduce_sum(tf.square(LTA), (0, 1)), (N, P))
    return fmean, fvar


def fully_correlated_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.
    :param Kmn: LM x N x P
    :param Kmm: LM x LM
    :param Knn: N x P or N x P x N x P
    :param f: data matrix, LM x 1
    :param q_sqrt: 1 x LM x LM  or 1 x ML
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: N x P
        - variance: N x P, N x P x P, P x N x N, N x P x N x P
    """
    m, v = fully_correlated_conditional_repeat(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                               full_output_cov=full_output_cov, q_sqrt=q_sqrt, white=white)
    return m[0, ...], v[0, ...]


def fully_correlated_conditional_repeat(Kmn, Kmm, Knn, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                                        white=False):
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.

    Note: This conditional can handle 'repetitions' R, given in `f` and `q_sqrt`.

    :param Kmn: LM x N x P
    :param Kmm: LM x LM
    :param Knn: N x P or N x P x N x P
    :param f: data matrix, LM x R
    :param q_sqrt: R x LM x LM  or R x ML
    :param full_cov: calculate covariance between inputs
    :param full_output_cov: calculate covariance between outputs
    :param white: use whitened representation
    :return:
        - mean: R x N x P
        - variance: R x N x P, R x N x P x P, R x P x N x N, R x N x P x N x P
    """
    logger.debug("fully correlated conditional")
    R = tf.shape(f)[1]
    M, N, K = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    # Lm: M x M    Kmn: M x NK
    Kmn = tf.reshape(Kmn, (M, N * K))  # M x NK
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # M x NK
    Ar = tf.reshape(A, (M, N, K))

    # compute the covariance due to the conditioning
    if full_cov and full_output_cov:
        # fvar = Knn - tf.matmul(Ar, Ar, transpose_a=True)  # NK x NK, then reshape?
        fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # N x K x N x K
    elif full_cov and not full_output_cov:
        At = tf.transpose(Ar)  # K x N x M
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # K x N x N
    elif not full_cov and full_output_cov:
        # This transpose is annoying
        At = tf.transpose(Ar, [1, 0, 2])  # N x M x K
        # fvar = Knn - tf.einsum('mnk,mnl->nkl', Ar, Ar)
        fvar = Knn - tf.matmul(At, At, transpose_a=True)  # N x K x K
    elif not full_cov and not full_output_cov:
        # Knn: N x K
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, K))  # Can also do this with a matmul

    # another backsubstitution in the unwhitened case
    if not white:
        # A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)  # M x NK
        raise NotImplementedError("Need to verify this.")  # pragma: no cover

    # f: M x R
    fmean = tf.matmul(f, A, transpose_a=True)  # R x M  *  M x NK  ->  R x NK
    fmean = tf.reshape(fmean, (R, N, K))  # R x N x K

    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M
        if q_sqrt.shape.ndims == 3:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # R x M x NK
            LTA = tf.matmul(Lf, A_tiled, transpose_a=True)  # R x M x NK
        elif q_sqrt.shape.ndims == 2:  # pragma: no cover
            raise NotImplementedError("Does not support diagonal q_sqrt yet...")
        else:  # pragma: no cover
            raise ValueError(f"Bad dimension for q_sqrt: {q_sqrt.shape.ndims}")

        if full_cov and full_output_cov:
            addvar = tf.matmul(LTA, LTA, transpose_a=True)  # R x NK x NK
            fvar = fvar[None, :, :, :, :] + tf.reshape(addvar, (R, N, K, N, K))
        elif full_cov and not full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, [R, M, N, K]), [0, 3, 1, 2])  # R x K x M x N
            addvar = tf.matmul(LTAr, LTAr, transpose_a=True)  # R x K x N x N
            fvar = fvar[None, ...] + addvar  # R x K x N x N
        elif not full_cov and full_output_cov:
            LTAr = tf.transpose(tf.reshape(LTA, (R, M, N, K)), [0, 2, 3, 1])  # R x N x K x M
            fvar = fvar[None, ...] + tf.matmul(LTAr, LTAr, transpose_b=True)  # R x N x K x K
        elif not full_cov and not full_output_cov:
            addvar = tf.reshape(tf.reduce_sum(LTA ** 2, axis=1), (R, N, K))  # R x N x K
            fvar = fvar[None, ...] + addvar  # R x N x K
    return fmean, fvar