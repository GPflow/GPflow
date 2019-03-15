# flake8: ignore=F811
# noqa: ignore=F811
# flake8: F811
# noqa: F811

import tensorflow as tf

from ..covariances import Kuf, Kuu
from ..features import (InducingPoints, MixedKernelSharedMof,
                        SeparateIndependentMof, SharedIndependentMof)
from ..kernels import (Combination, Mok, SeparateIndependentMok,
                       SeparateMixedMok, SharedIndependentMok)
from ..util import create_logger, default_jitter
from .dispatch import conditional
from .util import (base_conditional, expand_independent_outputs,
                   fully_correlated_conditional,
                   independent_interdomain_conditional)

logger = create_logger()


@conditional.register(object, SharedIndependentMof, SharedIndependentMok, object)
def _conditional(Xnew, feature, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                 white=False):
    """
    Multioutput conditional for an independent kernel and shared inducing features.
    Same behaviour as conditional with non-multioutput kernels.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, M]
    - Kuf: [M, N]
    - Kff: N or [N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param Xnew: data matrix, size [N, D].
    :param f: data matrix, [M, P]
    :param full_cov: return the covariance between the datapoints
    :param full_output_cov: return the covariance between the outputs.
        Note: as we are using a independent kernel these covariances will be zero.
    :param q_sqrt: matrix of standard-deviations or Cholesky matrices,
        size [M, P] or [P, M, M].
    :param white: boolean of whether to use the whitened representation
    :return:
        - mean:     [N, P]
        - variance: [N, P], [P, N, N], [N, P, P] or [N, P, N, P]
        Please see `gpflow.conditional._expand_independent_outputs` for more information
        about the shape of the variance, depending on `full_cov` and `full_output_cov`.
    """
    logger.debug("Conditional: SharedIndependentMof - SharedIndepedentMok")

    Kmm = Kuu(feature, kernel, jitter=default_jitter())  # [M, M]
    Kmn = Kuf(feature, kernel, Xnew)  # [M, N]
    if full_cov:
        Knn = kernel(Xnew, full_output_cov=False)[0, ...]  # [N, N]
    else:
        Knn = kernel(Xnew, full_output_cov=False)[..., 0]  # N

    fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt,
                                   white=white)  # [N, P],  [P, N, N] or [N, P]
    return fmean, expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, SeparateIndependentMof, SeparateIndependentMok, object)
@conditional.register(object, SharedIndependentMof, SeparateIndependentMok, object)
@conditional.register(object, SeparateIndependentMof, SharedIndependentMok, object)
def _conditional(Xnew, feature, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                 white=False):
    """
    Multi-output GP with independent GP priors.
    Number of latent processes equals the number of outputs (L = P).

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [P, M, M]
    - Kuf: [P, M, N]
    - Kff: [P, N] or [P, N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    - See above for the parameters and the return value.
    """

    logger.debug("conditional: object, SharedIndependentMof, SeparateIndependentMok, object")
    # Following are: [P, M, M]  -  [P, M, N]  -  P x N(x N)
    Kmms = Kuu(feature, kernel, jitter=default_jitter())  # [P, M, M]
    Kmns = Kuf(feature, kernel, Xnew)  # [P, M, N]
    kern_list = kernel.kernels if isinstance(kernel, Combination) else [kernel.kern] * len(
        feature.features)
    Knns = tf.stack([k(Xnew, full=full_cov) for k in kern_list], axis=0)
    Knns = tf.transpose(Knns)[:, :, None] if len(Knns.shape) == 2 else Knns # [P, N, 1]
    fs = tf.transpose(f)[:, :, None]  # [P, M, 1]
    # [P, 1, M, M]  or  [P, M, 1]
    q_sqrts = tf.transpose(q_sqrt)[:, :, None] if len(q_sqrt.shape) == 2 else q_sqrt[:, None, :, :]

    def single_gp_conditional(Kmm, Kmn, Knn, f, q_sqrt):
        return base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt, white=white)

    dtypes = (Kmms.dtype, Kmms.dtype)
    # rmu, rvar = tf.map_fn(lambda x: single_gp_conditional(x[0], x[1], x[2], x[3], x[4]),
    #                       (Kmms, Kmns, Knns,fs, q_sqrts), dtype=dtypes)
    for i in range(Knns.shape[0]):
        single_gp_conditional(Kmms[i, :, :], Kmns[i, :, :], Knns[i, :], fs[i, :, :], q_sqrts[i,
                                                                                      :, :])
    # [P,
    # N, 1], [P, 1, N, N] or [P, N, 1]
    fmu = tf.linalg.transpose(rmu[:, :, 0])  # [N, P]

    if full_cov:
        fvar = rvar[:, 0, :, :]  # [P, N, N]
    else:
        fvar = tf.transpose(rvar[..., 0])  # [N, P]

    return fmu, expand_independent_outputs(fvar, full_cov, full_output_cov)


@conditional.register(object, (SharedIndependentMof, SeparateIndependentMof), SeparateMixedMok,
                      object)
def _conditional(Xnew, feature, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                 white=False):
    """
    Interdomain conditional with independent latents.
    In this case the number of latent GPs (L) will be different than the number of outputs (P)

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], N x P

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.
    - See above for the parameters and the return value.
    """

    logger.debug("Conditional: (SharedIndependentMof, SeparateIndepedentMof) - SeparateMixedMok")
    Kmm = Kuu(feature, kernel, jitter=settings.numerics.jitter_level)  # [L, M, M]
    Kmn = Kuf(feature, kernel, Xnew)  # [M, L, N, P]
    Knn = kernel(Xnew, full_output_cov=full_output_cov) if full_cov \
        else kernel(Xnew, full_output_cov=full_output_cov)  # N x P(x N)x P  or  N x P(x P)

    return independent_interdomain_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                               full_output_cov=full_output_cov,
                                               q_sqrt=q_sqrt, white=white)


@conditional.register(object, InducingPoints, Mok, object)
def _conditional(Xnew, feature, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                 white=False):
    """
    Multi-output GP with fully correlated inducing variables.
    The inducing variables are shaped in the same way as evaluations of K, to allow a default
    inducing point scheme for multi-output kernels.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [M, L, M, L]
    - Kuf: [M, L, N, P]
    - Kff: [N, P, N, P], [N, P, P], N x P

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    Parameters
    ----------
    :param f: variational mean, ML x 1
    :param q_sqrt: standard-deviations or cholesky, ML x 1  or  [1, ML, ML]
    """
    logger.debug("Conditional: InducingPoints -- Mok")
    Kmm = Kuu(feature, kernel, jitter=settings.numerics.jitter_level)  # [M, L, M, L]
    Kmn = Kuf(feature, kernel, Xnew)  # [M, L, N, P]
    Knn = kernel(Xnew, full_output_cov=full_output_cov) if full_cov \
        else kernel(Xnew, full_output_cov=full_output_cov)  # N x P(x N)x P  or  N x P(x P)

    M, L, N, K = [Kmn.shape[i] for i in range(len(Kmn.shape))]
    Kmm = tf.reshape(Kmm, (M * L, M * L))

    if full_cov == full_output_cov:
        Kmn = tf.reshape(Kmn, (M * L, N * K))
        Knn = tf.reshape(Knn, (N * K, N * K)) if full_cov else tf.reshape(Knn, (N * K,))
        fmean, fvar = base_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov, q_sqrt=q_sqrt,
                                       white=white)  # NK x 1, 1 x NK(x NK)
        fmean = tf.reshape(fmean, (N, K))
        fvar = tf.reshape(fvar, (N, K, N, K) if full_cov else (N, K))
    else:
        Kmn = tf.reshape(Kmn, (M * L, N, K))
        fmean, fvar = fully_correlated_conditional(Kmn, Kmm, Knn, f, full_cov=full_cov,
                                                   full_output_cov=full_output_cov, q_sqrt=q_sqrt,
                                                   white=white)
    return fmean, fvar


@conditional.register(object, MixedKernelSharedMof, SeparateMixedMok, object)
def _conditional(Xnew, feature, kernel, f, *, full_cov=False, full_output_cov=False, q_sqrt=None,
                 white=False):
    """
    Most efficient routine to project L independent latent gps through a mixing matrix W.
    The mixing matrix is a member of the `SeparateMixedMok` and has shape P x L.

    The covariance matrices used to calculate the conditional have the following shape:
    - Kuu: [L, M, M]
    - Kuf: [L, M, N]
    - Kff: L x N or [L, N, N]

    Further reference
    -----------------
    - See `gpflow.conditionals._conditional` for a detailed explanation of
      conditional in the single-output case.
    - See the multiouput notebook for more information about the multiouput framework.

    """
    logger.debug("conditional: MixedKernelSharedMof, SeparateMixedMok")
    independent_cond = conditional.dispatch(object, SeparateIndependentMof, SeparateIndependentMok,
                                            object)
    gmu, gvar = independent_cond(Xnew, feature, kernel, f, full_cov=full_cov, q_sqrt=q_sqrt,
                                 full_output_cov=False, white=white)  # N x L, [L, N, N] or N x L

    gmu = tf.linalg.transpose(gmu)  # L x N
    if not full_cov:
        gvar = tf.linalg.transpose(gvar)  # L x N (x N)

    Wgmu = tf.tensordot(gmu, kernel.W(), [[0], [1]])  # N x P

    if full_output_cov:
        Wt_expanded = tf.linalg.transpose(kernel.W())[:, None, :]  # [L, 1, P]
        if full_cov:
            Wt_expanded = tf.expand_dims(Wt_expanded, axis=-1)  # [L, 1, P, 1]

        gvarW = tf.expand_dims(gvar, axis=2) * Wt_expanded  # [L, N, P] (x N)
        WgvarW = tf.tensordot(gvarW, kernel.W(), [[0], [1]])  # N x P (x N) x P
    else:
        if not full_cov:
            WgvarW = tf.tensordot(gvar, kernel.W() ** 2, [[0], [1]])  # N x P
        else:
            WgvarW = tf.tensordot(kernel.W() ** 2, gvar, [[1], [0]])  # P x N (x N)

    return Wgmu, WgvarW
