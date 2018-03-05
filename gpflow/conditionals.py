# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews
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

from . import settings, mean_functions
from .core.errors import GPflowError
from .decors import name_scope
from .dispatch import dispatch
from .expectations import expectation
from .features import InducingPoints, InducingFeature
from .kernels import Kernel
from .multikernels import MultiKernel, IndependentMultiKernel, IndependentFeature, MultiInducingPoints
from .multikernels import MixedMulti, MixedMultiIndependentFeature
from .probability_distributions import Gaussian


# TODO: Make all outputs of conditionals equal
# TODO: Add tensorflow assertions of shapes
# TODO: Remove `conditional()`?
# TODO: Ensure that R is handled correctly in all cases
# TODO: Should there be consistentcy between fmean out, and f in?
# Shapes to keep constant:
#  - f      : M x L x R  or M x L  or  M x R
#  - q_sqrt :


@dispatch(InducingFeature, Kernel, object, object)
@name_scope()
def feature_conditional(feat, kern, Xnew, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)
    Kmn = feat.Kuf(kern, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # N x R,  N x R


@dispatch(MixedMultiIndependentFeature, MixedMulti, object, object)
@name_scope()
def feature_conditional(feat, kern, Xnew, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmn = tf.stack([kern.K(Xnew, feat.Z) for kern in kern.kern_list], axis=0)  # L x N x M (TODO)

    Knn = tf.stack([kern.K(Xnew) if full_cov else kern.Kdiag(Xnew) for kern in kern.kern_list], axis=0)
    f = tf.transpose(f, [1, 0, 2])  # L x M x R
    q_sqrt = tf.transpose(q_sqrt, [1, 0, 2, 3])  # L x R x M x M

    def single_gp_conditional(t):
        Kmm, Kmn, Knn, f, q_sqrt = t
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    gmu, gvar = tf.map_fn(single_gp_conditional,
                          (Kmm, Kmn, Knn, f, q_sqrt),
                          (settings.float_type, settings.float_type))  # L x N x R  ,  L x N(x N)x R

    # Pmu = tf.matmul(self.P, gmu)  #  N x P
    Pmu = tf.tensordot(self.P, gmu, [[1], [0]])  # P x N x R

    PvarP = tf.tensordot(self.P, gvar, [[1], [0]])

    # f_mu:     P x N x R
    # f_var:    P (x P) x N (x N) x R  // P x N x P x N x R

    if full_cov:
        gvarP = gvar[None, ...] * self.P[..., None, None, None]  # P x L x N x N2 x R
    else:
        gvarP = gvar[None, ...] * self.P[..., None, None]  # P x L x N x N2
    if full_cov_output:
        # N = tf.shape(gmu)[0]
            # nP = tf.tile(self.P[None, :, :], [N, 1, 1])  # N,D_in,D_out

        # varP = tf.expand_dims(var, -1) * self.P[None, :, :]
        # varP = tf.expand_dims(gvar, -1) * nP

        # var is ND or NND
        # nP is N,D_in,D_out or N,N,D_in,D_out
        # PvarP is N,D_out,D_out or N,N,D_out,D_out
        # PvarP = tf.matmul(nP, varP, transpose_a=True)
        PvarP = tf.tensordot(self.P, gvarP, [[1], [1]])  # P x P x N x N x R

    else:
        # PvarP = tf.reduce_sum(self.P[None, :, :] * varP, 1)  # N,D_out
        # PvarP = tf.reduce_sum(nP * varP, -2)  # N,D_out
        if full_cov is True:
            raise NotImplementedError
#                     P2 = tf.expand_dims(self.P**2, [0, 1])
#                     PvarP = tf.reduce_sum(P2 * tf.expand_dims(var, -1), -2)
#                     PvarP = tf.matmul(tf.reshape(var, [N**2, D]))

        else:
            # var is N,D_in or N,N,D_in
            # P2 is 1,D_in,D_out or 1,1,D_in,D_out
            PvarP = tf.matmul(var, self.P**2)
    return Pmu, PvarP   


@dispatch(IndependentFeature, IndependentMultiKernel, object, object)
@name_scope()
def feature_conditional(feat, kern, Xnew, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors
    :param Xnew:
    :param feat:
    :param kern:
    :param f: M x L x R  or  M x L x 1
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: R x P x M  or R x P x M x M
    :param white:
    :return:
    """
    if f.shape.ndims != 3 or q_sqrt.shape.ndims != 4:
        raise GPflowError("IndependentFeature & IndependentMultiKernel combination requires separated GP posterior "
                          "representations, i.e. f: R x M x P.")

    # Following are: P x M x M  -  P x M x N  -  P x N(x N)
    Kmms = feat.Kuu(kern, jitter=settings.jitter)
    Kmns = feat.Kuf(kern, Xnew)
    Knns = tf.stack([kern.K(Xnew) if full_cov else kern.Kdiag(Xnew) for kern in kern.kern_list], axis=0)
    fs = tf.transpose(f, [1, 0, 2])  # P x M x R
    q_sqrts = tf.transpose(q_sqrt, [1, 0, 2, 3])

    def single_gp_conditional(t):
        Kmm, Kmn, Knn, f, q_sqrt = t
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    rmu, rvar = tf.map_fn(single_gp_conditional,
                          (Kmms, Kmns, Knns, fs, q_sqrts),
                          (settings.float_type, settings.float_type))  # P x N x R  ,  P x N(x N)x R

    rmu = tf.transpose(rmu[:, :, 0])
    rvar = tf.transpose(rvar[:, :, 0])
    return rmu, rvar


@dispatch(IndependentFeature, MultiKernel, object, object)
@name_scope()
def feature_conditional(feat, kern, Xnew, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors
    :param Xnew:
    :param feat:
    :param kern:
    :param f: M x L x R  or  M x L x 1
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: R x L x M  or R x L x M x M
    :param white:
    :return:
    """
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmn = feat.Kuf(kern, Xnew)  # M x L x N x K
    Knn = kern.K(Xnew, full_cov_output=full_cov_output) if full_cov \
        else kern.Kdiag(Xnew, full_cov_output=full_cov_output)  # N x K(x N)x K  or  N x K(x K)

    Kmm = tf.transpose(Kmm, [1, 2, 0])

    print(Knn.shape)

    return dependent_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, full_cov_output=full_cov_output, q_sqrt=q_sqrt,
                                 white=white)


@dispatch(MultiInducingPoints, MultiKernel, object, object)
@name_scope()
def feature_conditional(feat, kern, Xnew, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors
    :param f: M x R
    :param q_sqrt: R x M  or R x M x M
    """
    Kmm = feat.Kuu(kern, jitter=settings.numerics.jitter_level)  # M x L x M x K
    Kmn = feat.Kuf(kern, Xnew)  # M x L x N x K
    Knn = kern.K(Xnew, full_cov_output=full_cov_output) if full_cov \
        else kern.Kdiag(Xnew, full_cov_output=full_cov_output)  # N x K(x N)x K  or  N x K(x K)

    M, L, N, K = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]
    Kmm = tf.reshape(Kmm, (M * L, M * L))

    if full_cov == full_cov_output:
        Kmn = tf.reshape(Kmn, (M * L, N * K))
        Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
        fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # NK, NK(x NK)
        fmean = tf.reshape(fmean, (N, K))
        fvar = tf.reshape(fvar, (N, K, N, K) if full_cov else (N, K))
    else:
        Kmn = tf.reshape(Kmn, (M * L, N, K))
        fmean, fvar = fully_correlated_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, full_cov_output=full_cov_output,
                                                   q_sqrt=q_sqrt, white=white)
    return fmean, fvar


@name_scope()
def conditional(Xnew, X, kern, f, *, full_cov=False, q_sqrt=None, white=False):
    """
    Given f, representing the GP at the points X, produce the mean and
    (co-)variance of the GP at the points Xnew.

    Additionally, there may be Gaussian uncertainty about f as represented by
    q_sqrt. In this case `f` represents the mean of the distribution and
    q_sqrt the square-root of the covariance.

    Additionally, the GP may have been centered (whitened) so that
        p(v) = N(0, I)
        f = L v
    thus
        p(f) = N(0, LL^T) = N(0, K).
    In this case `f` represents the values taken by v.

    The method can either return the diagonals of the covariance matrix for
    each output (default) or the full covariance matrix (full_cov=True).

    We assume K independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

    :param Xnew: data matrix, size N x D.
    :param X: data points, size M x D.
    :param kern: GPflow kernel.
    :param f: data matrix, M x K, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x K or K x M x M.
    :param white: boolean of whether to use the whitened representation as
        described above.

    :return: two element tuple with conditional mean and variance.
    """
    num_data = tf.shape(X)[0]  # M
    Kmm = kern.K(X) + tf.eye(num_data, dtype=settings.float_type) * settings.numerics.jitter_level
    Kmn = kern.K(X, Xnew)
    if full_cov:
        Knn = kern.K(Xnew)
    else:
        Knn = kern.Kdiag(Xnew)
    return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


@name_scope()
def base_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, q_sqrt=None, white=False):
    # compute kernel stuff
    num_func = tf.shape(f)[1]  # R
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)

    # compute the covariance due to the conditioning
    if full_cov:
        fvar = Knn - tf.matmul(A, A, transpose_a=True)
        shape = tf.stack([num_func, 1, 1])
    else:
        fvar = Knn - tf.reduce_sum(tf.square(A), 0)
        shape = tf.stack([num_func, 1])
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # K x N x N or K x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # K x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # K x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # K x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # K x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # K x N
    fvar = tf.transpose(fvar)  # N x K or N x N x K

    return fmean, fvar


def independent_latents_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_cov_output=False, q_sqrt=None,
                                    white=False):
    """
    :param Kmn: L x M x N x P
    :param Kmm: L x M x M
    :param Knn: N x P  or  N x N  or  P x N x N  or  N x P x N x P
    :param f: data matrix, M x L x R
    :param q_sqrt: R x L x M x M  or  M x L x R
    :return: N x R x P  ,  N x R x P x P
    """
    # TODO: Allow broadcasting over L if priors are shared?
    R = tf.shape(f)[2]
    L, M, N, P = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]

    Lm = tf.cholesky(Kmm)  # L x M x M

    # Compute the projection matrix A
    Kmn = tf.reshape(Kmn, (L, M, N * P))
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # L x M x M  *  L x M x NP  ->  L x M x NP
    Ar = tf.reshape(A, (L, M, N, P))

    # compute the covariance due to the conditioning
    if full_cov and full_cov_output:
        fvar = Knn - tf.tensordot(Ar, Ar, [[0, 1], [0, 1]])  # N x P x N x P
    elif full_cov and not full_cov_output:
        At = tf.reshape(tf.transpose(Ar), (P, N, M * L))  # P x N x ML
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # P x N x N
    elif not full_cov and full_cov_output:
        At = tf.reshape(tf.transpose(Ar, [2, 3, 1, 0]), (N, P, M * L))  # N x P x ML
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # N x P x P
    elif not full_cov and not full_cov_output:
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, P))  # Knn: N x P

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(Lm, Ar)  # L x M x M  *  L x M x NP  ->  L x M x NP
        Ar = tf.reshape(A, (L, M, N, P))

    # mean: N x R x P
    fmean = tf.tensordot(f, Ar, [[0, 1], [1, 0]])  # R x NP
    fmean = tf.transpose(tf.reshape(fmean, (R, N, P)), [1, 0, 2])  # N x R x P

    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # R x L x M x M
        if q_sqrt.shape.ndims == 4:
            # Broadcast over L?
            A_tiled = tf.tile(A[None, :, :, :], tf.stack([R, 1, 1, 1]))  # R x L x M x NP
            LTA = tf.matmul(Lf, A_tiled, transpose_a=True)  # R x L x M x M  *  R x L x M x NP  ->  R x L x M x NP
        else:
            raise NotImplementedError()

        if full_cov and full_cov_output:
            LTAr = tf.reshape(LTA, (R, L * M, N * P))
            fvar = fvar[None, :, :, :, :] + tf.reshape(tf.matmul(LTAr, LTAr, transpose_a=True), (R, N, P, N, P))
        elif full_cov and not full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (R, L * M, N, P)), [0, 3, 1, 2])  # R x P x LM x N
            fvar = fvar[None, :, :, :] + tf.matmul(LTAr, LTAr, transpose_a=True)  # R x P x N x N
        elif not full_cov and full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (R, L * M, N, P)), [2, 0, 1, 3])  # N x R x LM x P
            fvar = fvar[:, None, :, :] + tf.matmul(LTAr, LTAr, transpose_a=True)  # N x R x P x P
        elif not full_cov and not full_cov_output:
            # N x R x P
            fvar = fvar[None, :, :] + tf.reshape(tf.reduce_sum(tf.square(LTA), (1, 2)), (R, N, P))
            fvar = tf.transpose(fvar, (N, R, P))
    return fmean, fvar


def fully_correlated_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    This function handles conditioning of multi-output GPs in the case where the conditioning
    points are all fully correlated, in both the prior and posterior.
    :param Kmn: M x N x K
    :param Kmm: M x M
    :param Knn: N x K  or  N x N  or  K x N x N  or  N x K x N x K
    :param f: data matrix, M x R
    :param q_sqrt: R x M x M  or  R x M
    :return: N x R x K  ,  N x R x K x K
    """
    R = tf.shape(f)[1]
    M, N, K = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]
    Lm = tf.cholesky(Kmm)

    # Compute the projection matrix A
    # Lm: M x M    Kmn: M x NK
    Kmn = tf.reshape(Kmn, (M, N * K))  # M x NK
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # M x NK
    Ar = tf.reshape(A, (M, N, K))

    # compute the covariance due to the conditioning
    if full_cov and full_cov_output:
        # fvar = Knn - tf.matmul(Ar, Ar, transpose_a=True)  # NK x NK, then reshape?
        fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # N x K x N x K
    elif full_cov and not full_cov_output:
        At = tf.transpose(Ar)  # K x N x M
        fvar = Knn - tf.matmul(At, At, transpose_b=True)  # K x N x N
    elif not full_cov and full_cov_output:
        # This transpose is annoying
        At = tf.transpose(Ar, [1, 0, 2])  # N x M x K
        # fvar = Knn - tf.einsum('mnk,mnl->nkl', Ar, Ar)
        fvar = Knn - tf.matmul(At, At, transpose_a=True)  # N x K x K
    elif not full_cov and not full_cov_output:
        # Knn: N x K
        fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, K))  # Can also do this with a matmul

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)  # M x NK
        raise NotImplementedError("Need to verify this.")

    # f: M x R
    fmean = tf.matmul(f, A, transpose_a=True)  # R x M  *  M x NK  ->  R x NK
    fmean = tf.reshape(fmean, (R, N, K))

    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M
        if q_sqrt.get_shape().ndims == 3:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # R x M x NK
            LTA = tf.matmul(Lf, A_tiled, transpose_a=True)  # R x M x NK
        elif q_sqrt.get_shape().ndims == 2:
            raise NotImplementedError("Does not support diagonal q_sqrt yet...")
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))

        if full_cov and full_cov_output:
            addvar = tf.matmul(LTA, LTA, transpose_a=True)  # R x NK x NK
            fvar = fvar[None, :, :, :, :] + tf.reshape(addvar, (R, N, K, N, K))
        elif full_cov and not full_cov_output:
            raise NotImplementedError()
        elif not full_cov and full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (R, M, N, K)), [2, 0, 3, 1])  # N x R x K x M
            fvar = fvar[:, None, :, :] + tf.matmul(LTAr, LTAr, transpose_b=True)  # N x R x K x K
        elif not full_cov and not full_cov_output:
            addvar = tf.reshape(tf.reduce_sum(tf.square(LTA), 1), (R, N, K))
            fvar = fvar[:, None, :] + tf.transpose(addvar, (1, 0, 2))  # N x R x K
    return fmean, fvar


def dependent_conditional(Kmn, Kmm, Knn, f, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    This function handles conditioning of single and multiple output GPs in
    various situations.
    :param Kmn: M x L x N x K
    :param Kmm: M x M  or  K x M x M
    :param Knn: N x K  or  N x N  or  K x N x N  or  N x K x N x K
    :param f: data matrix, M x L x R
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: R x L x M x M or R x L x M
    :param white:
    :return: N x R x K, N x R x K x K
    """

    # f: M x L x R
    if f.shape.ndims == 2:
        f = f[:, None, :]

    # Output dim Kmn: L1 x M x N x K
    if Kmn.shape.ndims == 2:  # Input: M x N
        N = tf.shape(Kmn)[1]
        Kmn = Kmn[None, :, :, None]
    elif Kmn.shape.ndims == 3:  # Input: M x N x K
        N = tf.shape(Kmn)[1]
        Kmn = Kmn[None, :, :, :]
    elif Kmn.shape.ndims == 4:  # Input: M x L1 x N x K
        N = tf.shape(Kmn)[2]
        if Kmm.shape.ndims != 4:
            # Case for Kmm.shape.ndims == 4 is handled later
            Kmn = tf.transpose(Kmn, [1, 0, 2, 3])
    else:  # pragma: no cover
        raise GPflowError("`Kmn` incompatible rank.")

    K = tf.shape(Kmn)[3]

    # Output dim Kmm: L1 x M x M
    if Kmm.shape.ndims == 2:  # M x M
        Kmm = Kmm[None, :, :]
    elif Kmm.shape.ndims == 3:  # M x M x L
        Kmm = tf.transpose(Kmm, [2, 0, 1])
    elif Kmm.shape.ndims == 4:  # M x L x M x L
        M, L = tf.shape(Kmm)[0], tf.shape(Kmm)[1]
        Kmm = tf.reshape(Kmm, (1, M * L, M * L))
        Kmn = tf.reshape(Kmn, (1, M * tf.shape(Kmn)[1], N, K))
    else:  # pragma: no cover
        raise GPflowError("`Kmm` incompatible rank (%i)." % Kmm.shape.ndims)
    Lm = tf.cholesky(Kmm)

    M = tf.shape(Kmn)[1]
    L = tf.shape(f)[1]
    R = tf.shape(f)[2]

    # Compute the projection matrix A
    # Lm: L x M x M    Kmn: L x M x NK
    # TODO: Need to sort out broadcasting.
    # We can have Lm needing broadcasting OR Kmn needing broadcasting. However, if both have L=1, then the GP can be
    # conditioned on the sum of the RVs WLOG.
    Kmn = tf.reshape(Kmn, (L, M, N * K))  # L x M x NK
    A = tf.matrix_triangular_solve(Lm, Kmn, lower=True)  # L x M x NK

    # compute the covariance due to the conditioning
    if full_cov:
        if full_cov_output:
            Ar = tf.reshape(A, (M * L, N, K))
            # fvar = Knn - tf.matmul(Ar, Ar, transpose_a=True)  # NK x NK
            fvar = Knn - tf.tensordot(Ar, Ar, [[0], [0]])  # N x K x N x K
        else:
            Ar = tf.transpose(tf.reshape(A, (M * L, N, K)))  # K x N x ML
            fvar = Knn - tf.matmul(Ar, Ar, transpose_b=True)  # K x N x N
    else:
        if full_cov_output:
            # Knn: N x K x K
            Ar = tf.transpose(tf.reshape(A, (M * L, N, K)), [1, 0, 2])  # N x ML x K
            # fvar = Knn - tf.einsum('mnk,mnl->nkl', Ar, Ar)
            fvar = Knn - tf.matmul(Ar, Ar, transpose_a=True)  # N x K x K
        else:
            # Knn: NK
            fvar = Knn - tf.reshape(tf.reduce_sum(tf.square(A), [0, 1]), (N, K))  # Can also do this with a matmul

    # another backsubstitution in the unwhitened case
    if not white:
        # if A.shape[0] == K, then Lm.shape[0] == K as well
        A = tf.matrix_triangular_solve(tf.matrix_transpose(Lm), A, lower=False)  # L x M x NK

    Almnk = tf.reshape(A, (L, M, N, K))
    # f: M x L x R
    # fmean = tf.tensordot(Ar, f, [[0, 1], [2, 1]])  # N x K x R
    fmean = tf.tensordot(f, Almnk, [[0, 1], [1, 0]])  # R x N x K
    fmean = tf.transpose(fmean, (1, 0, 2))  # N x R x K

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 3:
            # TODO: Check
            raise NotImplementedError()
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # L x M x NK
        elif q_sqrt.get_shape().ndims == 4:
            Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # R x L x M x M
            A_tiled = tf.tile(A[None, :, :, :], tf.stack([R, L // tf.shape(A)[0], 1, 1]))
            LTA = tf.matmul(Lf, A_tiled, transpose_a=True)  # R x L x M x M  *  R x L1 x M x NK  ->  R x L x M x NK
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            if full_cov_output:
                LTAr = tf.reshape(LTA, (L * M, N, K))  # ML x N x K
                fvar = fvar + tf.tensordot(LTAr, LTAr, [[0], [0]])  # N x K x N x K
            else:
                LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, K)))  # K x N x LM
                fvar = fvar + tf.matmul(LTAr, LTAr, transpose_b=True)  # K x N x N
        else:
            if full_cov_output:
                LTAr = tf.transpose(tf.reshape(LTA, (R, L * M, N, K)), [0, 2, 3, 1])  # R x N x K x LM
                fvar = fvar + tf.matmul(LTAr, LTAr, transpose_b=True)  # R x N x K x K
                fvar = tf.transpose(fvar, [1, 0, 2, 3])
            else:
                fvar = tf.Print(fvar, [tf.shape(fvar)])
                fvar = fvar + tf.reshape(tf.reduce_sum(tf.square(LTA), (1, 2)), (N, K))  # R x N x K
    fmean = tf.Print(fmean, [tf.shape(fmean)], message="fmean ")
    return fmean, fvar


@name_scope()
def uncertain_conditional(Xnew_mu, Xnew_var, feat, kern, q_mu, q_sqrt, *,
                          mean_function=None, full_cov_output=False, full_cov=False, white=False):
    """
    Calculates the conditional for uncertain inputs Xnew, p(Xnew) = N(Xnew_mu, Xnew_var).
    See ``conditional`` documentation for further reference.

    :param Xnew_mu: mean of the inputs, size N x Din
    :param Xnew_var: covariance matrix of the inputs, size N x Din x Din
    :param feat: gpflow.InducingFeature object, only InducingPoints is supported
    :param kern: gpflow kernel or ekernel object.
    :param q_mu: mean inducing points, size M x Dout
    :param q_sqrt: cholesky of the covariance matrix of the inducing points, size Dout x M x M
    :param full_cov_output: boolean wheter to compute covariance between output dimension.
                            Influences the shape of return value ``fvar``. Default is False
    :param white: boolean whether to use whitened representation. Default is False.

    :return fmean, fvar: mean and covariance of the conditional, size ``fmean`` is N x Dout,
            size ``fvar`` depends on ``full_cov_output``: if True ``f_var`` is N x Dout x Dout,
            if False then ``f_var`` is N x Dout
    """

    # TODO: Tensorflow 1.4 doesn't support broadcasting in``tf.matmul`` and
    # ``tf.matrix_triangular_solve``. This is reported in issue 216.
    # As a temporary workaround, we are using ``tf.einsum`` for the matrix
    # multiplications and tiling in the triangular solves.
    # The code that should be used once the bug is resolved is added in comments.

    if not isinstance(feat, InducingPoints):
        raise NotImplementedError

    if full_cov:
        # TODO: ``full_cov`` True would return a ``fvar`` of shape N x N x D x D,
        # encoding the covariance between input datapoints as well.
        # This is not implemented as this feature is only used for plotting purposes.
        raise NotImplementedError

    pXnew = Gaussian(Xnew_mu, Xnew_var)

    num_data = tf.shape(Xnew_mu)[0]  # number of new inputs (N)
    num_ind = tf.shape(q_mu)[0]  # number of inducing points (M)
    num_func = tf.shape(q_mu)[1]  # output dimension (D)

    q_sqrt_r = tf.matrix_band_part(q_sqrt, -1, 0)  # D x M x M

    eKuf = tf.transpose(expectation(pXnew, (kern, feat)))  # M x N (psi1)
    Kuu = feat.Kuu(kern, jitter=settings.numerics.jitter_level)  # M x M
    Luu = tf.cholesky(Kuu)  # M x M

    if not white:
        q_mu = tf.matrix_triangular_solve(Luu, q_mu, lower=True)
        Luu_tiled = tf.tile(Luu[None, :, :], [num_func, 1, 1])  # remove line once issue 216 is fixed
        q_sqrt_r = tf.matrix_triangular_solve(Luu_tiled, q_sqrt_r, lower=True)

    Li_eKuf = tf.matrix_triangular_solve(Luu, eKuf, lower=True)  # M x N
    fmean = tf.matmul(Li_eKuf, q_mu, transpose_a=True)

    eKff = expectation(pXnew, kern)  # N (psi0)
    eKuffu = expectation(pXnew, (kern, feat), (kern, feat))  # N x M x M (psi2)
    Luu_tiled = tf.tile(Luu[None, :, :], [num_data, 1, 1])  # remove this line, once issue 216 is fixed
    Li_eKuffu = tf.matrix_triangular_solve(Luu_tiled, eKuffu, lower=True)
    Li_eKuffu_Lit = tf.matrix_triangular_solve(Luu_tiled, tf.matrix_transpose(Li_eKuffu), lower=True)  # N x M x M
    cov = tf.matmul(q_sqrt_r, q_sqrt_r, transpose_b=True)  # D x M x M

    if mean_function is None or isinstance(mean_function, mean_functions.Zero):
        e_related_to_mean = tf.zeros((num_data, num_func, num_func), dtype=settings.float_type)
    else:
        # Update mean: \mu(x) + m(x)
        fmean = fmean + expectation(pXnew, mean_function)

        # Calculate: m(x) m(x)^T + m(x) \mu(x)^T + \mu(x) m(x)^T,
        # where m(x) is the mean_function and \mu(x) is fmean
        e_mean_mean = expectation(pXnew, mean_function, mean_function)  # N x D x D
        Lit_q_mu = tf.matrix_triangular_solve(Luu, q_mu, adjoint=True)
        e_mean_Kuf = expectation(pXnew, mean_function, (kern, feat))  # N x D x M
        # einsum isn't able to infer the rank of e_mean_Kuf, hence we explicitly set the rank of the tensor:
        e_mean_Kuf = tf.reshape(e_mean_Kuf, [num_data, num_func, num_ind])
        e_fmean_mean = tf.einsum("nqm,mz->nqz", e_mean_Kuf, Lit_q_mu)  # N x D x D
        e_related_to_mean = e_fmean_mean + tf.matrix_transpose(e_fmean_mean) + e_mean_mean

    if full_cov_output:
        fvar = (
                tf.matrix_diag(tf.tile((eKff - tf.trace(Li_eKuffu_Lit))[:, None], [1, num_func])) +
                tf.matrix_diag(tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov)) +
                # tf.matrix_diag(tf.trace(tf.matmul(Li_eKuffu_Lit, cov))) +
                tf.einsum("ig,nij,jh->ngh", q_mu, Li_eKuffu_Lit, q_mu) -
                # tf.matmul(q_mu, tf.matmul(Li_eKuffu_Lit, q_mu), transpose_a=True) -
                fmean[:, :, None] * fmean[:, None, :] +
                e_related_to_mean
        )
    else:
        fvar = (
                (eKff - tf.trace(Li_eKuffu_Lit))[:, None] +
                tf.einsum("nij,dji->nd", Li_eKuffu_Lit, cov) +
                tf.einsum("ig,nij,jg->ng", q_mu, Li_eKuffu_Lit, q_mu) -
                fmean ** 2 +
                tf.matrix_diag_part(e_related_to_mean)
        )

    return fmean, fvar
