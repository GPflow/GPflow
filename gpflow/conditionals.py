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
from .decors import name_scope
from .dispatch import dispatch 
from .expectations import expectation
from .features import InducingPoints, InducingFeature
from .kernels import Kernel, Combination
from .multikernels import Mok, SharedIndependentMok, SeparateIndependentMok, SeparateMixedMok
from .multikernels import Kuf, Kuu
from .multifeatures import Mof, SeparateIndependentMof, SharedIndependentMof, SeparateMixedMof
from .probability_distributions import Gaussian

# TODO: Make all outputs of conditionals equal
# TODO: Add tensorflow assertions of shapes
# TODO: Remove `conditional()`?
# Shapes to keep constant:
#  - f      : M x L x R  or M x L  or  M x R
#  - q_sqrt :

# TODO move implementations of conditional() for multi-output kernels to multioutput module?

# There's a lot of duplicate code in the various types of conditionals ... e.g.
# they all do L = cholesky(Kmm), A = L^-1 Lmn ... I think it'd be much cleaner
# & easier to understand if we break things up from the bottom up, e.g.
# something like get_A_and_fvar with multiple dispatch for the different
# combinations of feature & kernel to return the appropriately shaped objects,
# and then a single "general" conditional that calls these helper functions
# instead of doing all the nitty-gritty reshaping by itself.

"""
conditionals.py
The only thing that is completely specified for an implementation of `conditional()`, is the return
shapes.

Option 1:
--------
fmean : N x P
fvar  : N x P  or  N x P x P  or  P x N x N  or  N x P x N x P
This is occurs when the multi-output nature is described purely by repeating the same prior P
times. Full covariances over outputs will always be diagonal.

Option 2:
--------
fmean : N x P
fvar  : N x P  or  N x P x P  or  P x N x N  or  N x P x N x P
For when we have a truly multi-output kernel, with P the output dimension

Option 3:
--------
fmean : N x P
fvar  : N x P  or  N x P x P  or  P x N x N  or  N x P x N x P
Multi-output kernel, with ? repetitions of the prior.

In general, we should aim to keep the shapes of f, q_sqrt, Kuu, and Kuf consistent as well. For
optimisation reasons, however, this can be departed from. The standard is:
f      : M x L
q_sqrt : M x L  or  M x M  or L x M x M
"""


def expand_independent_outputs(fvar, full_cov, full_cov_output):
    # TODO point of this function?
    if not full_cov_output:
        # Output shape should be N x R or R x N x N, which it already is.
        return fvar
    elif full_cov:
        # Output shape should be ???
        raise NotImplementedError



@dispatch(object, InducingFeature, Kernel, object)
@dispatch(object, SharedIndependentMof, SharedIndependentMok, object)
@name_scope()
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Single-output GP allowing repetitions
    :param f: M x R
    :param q_sqrt: M x R  or  R x M x M
    :return: N x R  or R x N x N  or  N x R x R  or  N x R x N x R
    """
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x N
    if full_cov:
        Knn = kern.K(Xnew)  # N x N
    else:
        Knn = kern.Kdiag(Xnew)  # N
    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # N x R,  N x (x N) x R
    return fmean, expand_independent_outputs(fvar, full_cov, full_cov_output)


@dispatch(object, SharedIndependentMof, SharedIndependentMok, object)
@name_scope()
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    """
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x N
    if full_cov:
        Knn = kern.K(Xnew, full_cov_output=False)[..., 0]  # N x N
    else:
        Knn = kern.Kdiag(Xnew, full_cov_output=False)[..., 0]  # N
    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # N x P,  N x (x N) x P
    return fmean, expand_independent_outputs(fvar, full_cov, full_cov_output)


@dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
@name_scope()
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors.
    Number of latent processes equals the number of outputs (L = P). Expected kernels:
     Kmm
    :param f: M x P
    :param q_sqrt: M x P  or  P x M x M
    :return: N x P ,
    """
    print("Conditional")
    print("object, SharedIndependentMof, SeparateIndependentMok, object")
    print("object, SeparateIndependentMof, SharedIndependentMok, object")
    print("object, SeparateIndependentMof, SeparateIndependentMok, object")
    # Following are: P x M x M  -  P x M x N  -  P x N(x N)
    Kmms = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # P x M x M
    Kmns = Kuf(feat, kern, Xnew)  # P x M x N
    kern_list = kern.kern_list if isinstance(kern, Combination) else [kern.kern] * len(feat.feat_list)
    Knns = tf.stack([k.K(Xnew) if full_cov else k.Kdiag(Xnew) for k in kern_list], axis=0)
    fs = tf.transpose(f)[:, :, None]  # P x M x 1
    # P x 1 x M x M  or  P x M x 1
    q_sqrts = tf.transpose(q_sqrt)[:, :, None] if q_sqrt.shape.ndims == 2 else q_sqrt[:, None, :, :]

    def single_gp_conditional(t):
        Kmm, Kmn, Knn, f, q_sqrt = t
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    rmu, rvar = tf.map_fn(single_gp_conditional,
                          (Kmms, Kmns, Knns, fs, q_sqrts),
                          (settings.float_type, settings.float_type))  # P x N x 1  ,  P x N(x N) x 1
    
    fmu = tf.matrix_transpose(rmu[..., 0])
    fvar = rvar[..., 0]

    if full_cov_output and full_cov:
        fvar = tf.diag(tf.transpose(fvar, [1, 2, 0]))
        fvar = tf.transpose(fvar, [0, 2, 1, 3])
    elif not full_cov_output and full_cov:
        pass
    elif full_cov_output and not full_cov:
        fvar = tf.diag(tf.matrix_transpose(fvar))
    elif not full_cov_output and not full_cov:
        fvar = tf.matrix_transpose(fvar)

    return fmu, fvar


@dispatch(object, SharedIndependentMof, SeparateIndependentMok, object)
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    cond_impl = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
    return cond_impl(Xnew, feat, kern, f, full_cov_output=full_cov_output, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


@dispatch(object, SeparateIndependentMof, SharedIndependentMok, object)
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    cond_impl = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
    return cond_impl(Xnew, feat, kern, f, full_cov_output=full_cov_output, full_cov=full_cov, q_sqrt=q_sqrt, white=white)


@dispatch(object, (SharedIndependentMof, SeparateIndependentMof), SeparateMixedMok, object)
@name_scope()
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors
    :param Xnew:
    :param feat:
    :param kern:
    :param f: M x L
    :param full_cov:
    :param full_cov_output:
    :param q_sqrt: L x M  or L x M x M
    :param white:
    :return:
    """
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x K
    Knn = kern.K(Xnew, full_cov_output=full_cov_output) if full_cov \
        else kern.Kdiag(Xnew, full_cov_output=full_cov_output)  # N x K(x N)x K  or  N x K(x K)

    return independent_latents_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, full_cov_output=full_cov_output,
                                           q_sqrt=q_sqrt, white=white)


@dispatch(object, InducingPoints, Mok, object)
@name_scope()
def conditional(Xnew, feat, kern, f, *, full_cov=False, full_cov_output=False, q_sqrt=None, white=False):
    """
    Multi-output GP with fully correlated inducing variables.
    The inducing variables are shaped in the same way as evaluations of K, to allow a default
    inducing point scheme for multi-output kernels.

     Kmm : M x L x M x P
     Kmn : M x L x N x P

    :param f: ML x 1
    :param q_sqrt: ML x 1  or  1 x ML x ML
    """
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x L x M x P
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    Knn = kern.K(Xnew, full_cov_output=full_cov_output) if full_cov \
        else kern.Kdiag(Xnew, full_cov_output=full_cov_output)  # N x P(x N)x P  or  N x P(x P)

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
        # TODO: Fix this output shape
    fmean = tf.Print(fmean, [tf.shape(fmean), tf.shape(fvar)], summarize=100)
    return fmean, fvar


@dispatch(object, object, Kernel, object)  # TODO: Make types more specific to TensorFlow types?
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

    We assume R independent GPs, represented by the columns of f (and the
    last dimension of q_sqrt).

    :param Xnew: data matrix, size N x D.
    :param X: data points, size M x D.
    :param kern: GPflow kernel.
    :param f: data matrix, M x R, representing the function values at X,
        for K functions.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x R or R x M x M.
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
    :return: N x R  or N x N x R
    """
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
    fvar = tf.tile(tf.expand_dims(fvar, 0), shape)  # R x N x N or R x N

    # another backsubstitution in the unwhitened case
    if not white:
        A = tf.matrix_triangular_solve(tf.transpose(Lm), A, lower=False)

    # construct the conditional mean
    fmean = tf.matmul(A, f, transpose_a=True)

    if q_sqrt is not None:
        if q_sqrt.get_shape().ndims == 2:
            LTA = A * tf.expand_dims(tf.transpose(q_sqrt), 2)  # R x M x N
        elif q_sqrt.get_shape().ndims == 3:
            L = tf.matrix_band_part(q_sqrt, -1, 0)  # R x M x M
            A_tiled = tf.tile(tf.expand_dims(A, 0), tf.stack([num_func, 1, 1]))
            LTA = tf.matmul(L, A_tiled, transpose_a=True)  # R x M x N
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))
        if full_cov:
            fvar = fvar + tf.matmul(LTA, LTA, transpose_a=True)  # R x N x N
        else:
            fvar = fvar + tf.reduce_sum(tf.square(LTA), 1)  # R x N
    fvar = tf.transpose(fvar)  # N x R or N x N x R

    return fmean, fvar


def independent_latents_conditional(Kmn, Kmm, Knn, f, *, full_cov=False, full_cov_output=False, q_sqrt=None,
                                    white=False):
    """

    :param Kmn: M x L x N x P
    :param Kmm: L x M x M
    :param Knn: N x P  or  N x N  or  P x N x N  or  N x P x N x P
    :param f: data matrix, M x L
    :param q_sqrt: L x M x M  or  M x L
    :return: N x P  ,  N x R x P x P
    """
    # TODO: Allow broadcasting over L if priors are shared?
    # TODO: Change Kmn to be L x M x N x P? Saves a transpose...
    M, L, N, P = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]

    Lm = tf.cholesky(Kmm)  # L x M x M

    # Compute the projection matrix A
    Kmn = tf.reshape(tf.transpose(Kmn, (1, 0, 2, 3)), (L, M, N * P))
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

    fmean = tf.tensordot(Ar, f, [[0, 1], [0, 1]])  # N x P

    if q_sqrt is not None:
        Lf = tf.matrix_band_part(q_sqrt, -1, 0)  # L x M x M
        if q_sqrt.shape.ndims == 3:
            LTA = tf.matmul(Lf, A, transpose_a=True)  # L x M x M  *  L x M x NP  ->  L x M x NP
        else:
            raise NotImplementedError()

        if full_cov and full_cov_output:
            LTAr = tf.reshape(LTA, (L * M, N * P))
            fvar = fvar + tf.reshape(tf.matmul(LTAr, LTAr, transpose_a=True), (N, P, N, P))
        elif full_cov and not full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [0, 3, 1, 2])  # P x LM x N
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # P x N x N
        elif not full_cov and full_cov_output:
            LTAr = tf.transpose(tf.reshape(LTA, (L * M, N, P)), [1, 0, 2])  # N x LM x P
            fvar = fvar + tf.matmul(LTAr, LTAr, transpose_a=True)  # N x P x P
        elif not full_cov and not full_cov_output:
            fvar = fvar + tf.reshape(tf.reduce_sum(tf.square(LTA), (0, 1)), (N, P))
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
