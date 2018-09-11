# Copyright 2018 GPflow authors
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

from .features import SeparateIndependentMof, SharedIndependentMof, MixedKernelSharedMof, MixedKernelSeparateMof
from .features import Kuu, Kuf
from .kernels import Mok, SharedIndependentMok, SeparateIndependentMok, SeparateMixedMok
from .. import settings
from ..conditionals import base_conditional, _expand_independent_outputs, _sample_mvn
from ..decors import name_scope, params_as_tensors_for
from ..dispatch import conditional, sample_conditional
from ..features import InducingPoints
from ..kernels import Combination


logger = settings.logger()


# -----------
# Conditional
# -----------

@conditional.register(object, SharedIndependentMof, SharedIndependentMok, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Multioutput conditional for an independent kernel and shared inducing features.
    Same behaviour as conditional with non-multioutput kernels.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: M x M
    - Kuf: M x N
    - Kff: N or N x N

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size N x D.
    :param f: data matrix, M x P
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        Note: as we are using a independent kernel these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size M x P or P x M x M.
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     N x P
        - variance: N x P, P x N x N, N x P x P or N x P x N x P
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    logger.debug("Conditional: SharedIndependentMof - SharedIndepedentMok")


    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x N
    if full_cov:
        Knn = kern.K(Xnew, full_output_cov=False)[0, ...]  # N x N
    else:
        Knn = kern.Kdiag(Xnew, full_output_cov=False)[..., 0]  # N

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # N x P,  P x N x N or N x P
    return fmean, _expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, SeparateIndependentMof, SeparateIndependentMok, object)
@conditional.register(object, SharedIndependentMof, SeparateIndependentMok, object)
@conditional.register(object, SeparateIndependentMof, SharedIndependentMok, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Multi-output GP with independent GP priors.
    Number of latent processes equals the number of outputs (L = P).

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: P x M x M
    - Kuf: P x M x N
    - Kff: P x N or P x N x N

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    - See above for the parameters and the return value.
    """

    logger.debug("conditional: object, SharedIndependentMof, SeparateIndependentMok, object")
    # Following are: P x M x M  -  P x M x N  -  P x N(x N)
    Kmms = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # P x M x M
    Kmns = Kuf(feat, kern, Xnew)  # P x M x N
    kern_list = kern.kernels if isinstance(kern, Combination) else [kern.kern] * len(feat.feat_list)
    Knns = tf.stack([k.K(Xnew) if full_cov else k.Kdiag(Xnew) for k in kern_list], axis=0)
    fs = tf.transpose(f)[:, :, None]  # P x M x 1
    # P x 1 x M x M  or  P x M x 1
    q_sqrts = tf.transpose(q_sqrt)[:, :, None] if q_sqrt.shape.ndims == 2 else q_sqrt[:, None, :, :]

    def single_gp_conditional(t):
        Kmm, Kmn, Knn, f, q_sqrt = t
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    rmu, rvar = tf.map_fn(single_gp_conditional,
                          (Kmms, Kmns, Knns, fs, q_sqrts),
                          (settings.float_type, settings.float_type))  # P x N x 1, P x 1 x N x N or P x N x 1

    fmu = tf.matrix_transpose(rmu[:, :, 0])  # N x P

    if full_cov:
        fvar = rvar[:, 0, :, :]  # P x N x N
    else:
        fvar = tf.transpose(rvar[..., 0])  # N x P

    return fmu, _expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, (SharedIndependentMof, SeparateIndependentMof), SeparateMixedMok, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Interdomain conditional with independent latents.
    In this case the number of latent GPs (L) will be different than the number of outputs (P)

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: L x M x M
    - Kuf: M x L x N x P
    - Kff: N x P x N x P, N x P x P, N x P

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    - See above for the parameters and the return value.
    """

    logger.debug("Conditional: (SharedIndependentMof, SeparateIndepedentMof) - SeparateMixedMok")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # L x M x M
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    Knn = kern.K(Xnew, full_output_cov=full_output_cov) if full_cov \
        else kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P(x N)x P  or  N x P(x P)

    return independent_interdomain_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)


@conditional.register(object, InducingPoints, Mok, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Multi-output GP with fully correlated inducing variables.
    The inducing variables are shaped in the same way as evaluations of K, to allow a default
    inducing point scheme for multi-output kernels.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: M x L x M x L
    - Kuf: M x L x N x P
    - Kff: N x P x N x P, N x P x P, N x P

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param f: variational mean, ML x 1
    :param q_sqrt: standard-deviations or cholesky, ML x 1  or  1 x ML x ML
    """
    logger.debug("Conditional: InducingPoints -- Mok")
    Kmm = Kuu(feat, kern, jitter=settings.numerics.jitter_level)  # M x L x M x L
    Kmn = Kuf(feat, kern, Xnew)  # M x L x N x P
    Knn = kern.K(Xnew, full_output_cov=full_output_cov) if full_cov \
        else kern.Kdiag(Xnew, full_output_cov=full_output_cov)  # N x P(x N)x P  or  N x P(x P)

    M, L, N, K = [tf.shape(Kmn)[i] for i in range(Kmn.shape.ndims)]
    Kmm = tf.reshape(Kmm, (M * L, M * L))

    if full_cov == full_output_cov:
        Kmn = tf.reshape(Kmn, (M * L, N * K))
        Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
        fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)  # NK x 1, 1 x NK(x NK)
        fmean = tf.reshape(fmean, (N, K))
        fvar = tf.reshape(fvar, (N, K, N, K) if full_cov else (N, K))
    else:
        Kmn = tf.reshape(Kmn, (M * L, N, K))
        fmean, fvar = fully_correlated_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                                   full_output_cov=full_output_cov, q_sqrt=q_sqrt, white=white)
    return fmean, fvar


@conditional.register(object, (MixedKernelSharedMof,MixedKernelSeparateMof), SeparateMixedMok, object)
@name_scope("conditional")
def _conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False):
    """
    Most efficient routine to project L independent latent gps through a mixing matrix W.
    The mixing matrix is a member of the `SeparateMixedMok` and has shape P x L.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: L x M x M
    - Kuf: L x M x N
    - Kff: L x N or L x N x N

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    """
    logger.debug("conditional: (MixedKernelSharedMof, MixedKernelSeparateMof), SeparateMixedMok")
    independent_cond = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
    gmu, gvar = independent_cond(Xnew, feat, kern, f, full_cov=full_cov, q_sqrt=q_sqrt,
                                 full_output_cov=False, white=white)  # N x L, L x N x N or N x L

    gmu = tf.matrix_transpose(gmu)  # L x N
    if not full_cov:
        gvar = tf.matrix_transpose(gvar)  # L x N (x N)

    Wgmu = tf.tensordot(gmu, kern.W, [[0], [1]])  # N x P

    if full_output_cov:
        Wt_expanded = tf.matrix_transpose(kern.W)[:, None, :]  # L x 1 x P
        if full_cov:
            Wt_expanded = tf.expand_dims(Wt_expanded, axis=-1)  # L x 1 x P x 1

        gvarW = tf.expand_dims(gvar, axis=2) * Wt_expanded  # L x N x P (x N)
        WgvarW = tf.tensordot(gvarW, kern.W, [[0], [1]])  # N x P (x N) x P
    else:
        if not full_cov:
            WgvarW = tf.tensordot(gvar, kern.W ** 2, [[0], [1]])  # N x P
        else:
            WgvarW = tf.tensordot(kern.W ** 2, gvar, [[1], [0]])  # P x N (x N)

    return Wgmu, WgvarW


# ------------------
# Sample conditional
# ------------------

@sample_conditional.register(object, (MixedKernelSharedMof, MixedKernelSeparateMof), SeparateMixedMok, object)
@name_scope("sample_conditional")
def _sample_conditional(Xnew, feat, kern, f, *, full_cov=False, full_output_cov=False, q_sqrt=None, white=False, num_samples=None):
    """
    `sample_conditional` will return a sample from the conditinoal distribution.
    In most cases this means calculating the conditional mean m and variance v and then
    returning m + sqrt(v) * eps, with eps ~ N(0, 1).
    However, for some combinations of Mok and Mof more efficient sampling routines exists.
    The dispatcher will make sure that we use the most efficent one.

    :return: N x P (full_output_cov = False) or N x P x P (full_output_cov = True)
    """
    logger.debug("sample conditional: (MixedKernelSharedMof, MixedKernelSeparateMof), SeparateMixedMok")
    if full_cov:
        raise NotImplementedError("full_cov not yet implemented")
    if full_output_cov:
        raise NotImplementedError("full_output_cov not yet implemented")
    independent_cond = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok, object)
    g_mu, g_var = independent_cond(Xnew, feat, kern, f, white=white, q_sqrt=q_sqrt,
                                   full_output_cov=False, full_cov=False)  # N x L, N x L
    g_sample = _sample_mvn(g_mu, g_var, "diag", num_samples=num_samples)  # N x L
    with params_as_tensors_for(kern):
        f_sample = tf.einsum("pl,nl->np", kern.W, g_sample)
        f_mu = tf.einsum("pl,nl->np", kern.W, g_mu)
        # W g_var W.T
        # [P, L] @ [L, L] @ [L, P]
        # \sum_l,l' W_pl g_var_ll' W_p'l'
        # \sum_l W_pl g_var_nl W_p'l
        # -> 
        f_var = tf.einsum("pl,nl,pl->np", kern.W, g_var, kern.W)
    return f_sample, f_mu, f_var


# -----------------
# Conditional maths
# -----------------

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
        if q_sqrt.get_shape().ndims == 3:
            A_tiled = tf.tile(A[None, :, :], tf.stack([R, 1, 1]))  # R x M x NK
            LTA = tf.matmul(Lf, A_tiled, transpose_a=True)  # R x M x NK
        elif q_sqrt.get_shape().ndims == 2:  # pragma: no cover
            raise NotImplementedError("Does not support diagonal q_sqrt yet...")
        else:  # pragma: no cover
            raise ValueError("Bad dimension for q_sqrt: %s" %
                             str(q_sqrt.get_shape().ndims))

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
            addvar = tf.reshape(tf.reduce_sum(tf.square(LTA), axis=1), (R, N, K))  # R x N x K
            fvar = fvar[None, ...] + addvar  # R x N x K
    return fmean, fvar
