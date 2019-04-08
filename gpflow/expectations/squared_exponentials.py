import tensorflow as tf

from . import dispatch
from .. import kernels
from .. import mean_functions as mfn
from ..features import InducingPoints
from ..probability_distributions import (DiagonalGaussian, Gaussian,
                                         MarkovGaussian)
from ..util import NoneType, default_float
from .expectations import expectation


@dispatch.expectation.register(Gaussian, kernels.RBF, NoneType, NoneType, NoneType)
def _E(p, kern, _, __, ___, nghp=None):
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: RBF kernel

    :return: N
    """
    return kern(p.mu, full=False)


@dispatch.expectation.register(Gaussian, kernels.RBF, InducingPoints, NoneType, NoneType)
def _E(p, kern, feat, _, __, nghp=None):
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: RBF kernel

    :return: NxM
    """
    # use only active dimensions
    Xcov = kern.slice_cov(p.cov)
    Z, Xmu = kern.slice(feat.Z, p.mu)
    D = Xmu.shape[1]

    lengthscale = kern.lengthscale
    if not kern.ard:
        lengthscale = tf.zeros((D,), dtype=lengthscale.dtype) + kern.lengthscale

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscale ** 2) + Xcov)  # NxDxD

    all_diffs = tf.transpose(Z) - tf.expand_dims(Xmu, 2)  # NxDxM
    exponent_mahalanobis = tf.linalg.triangular_solve(chol_L_plus_Xcov, all_diffs,
                                                      lower=True)  # NxDxM
    exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    sqrt_det_L = tf.reduce_prod(lengthscale)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1))
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    return kern.variance * (determinants[:, None] * exponent_mahalanobis)


@dispatch.expectation.register(Gaussian, mfn.Identity, NoneType, kernels.RBF, InducingPoints)
def _E(p, mean, _, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_n K_{x_n, Z}>_p(x_n)
        - K_{.,.} :: RBF kernel

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    D = Xmu.shape[1]

    lengthscale = kern.lengthscale
    if not kern.ard:
        lengthscale = tf.zeros((D,), dtype=lengthscale.dtype) + lengthscale

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscale ** 2) + Xcov)  # NxDxD
    all_diffs = tf.transpose(feat.Z) - tf.expand_dims(Xmu, 2)  # NxDxM

    sqrt_det_L = tf.reduce_prod(lengthscale)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1))
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    exponent_mahalanobis = tf.linalg.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
    non_exponent_term = tf.linalg.matmul(Xcov, exponent_mahalanobis, transpose_a=True)
    non_exponent_term = tf.expand_dims(Xmu, 2) + non_exponent_term  # NxDxM

    exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    return kern.variance * (determinants[:, None] *
                            exponent_mahalanobis)[:, None, :] * non_exponent_term


@dispatch.expectation.register(MarkovGaussian, mfn.Identity, NoneType, kernels.RBF, InducingPoints)
def _E(p, mean, _, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1})
        - K_{.,.} :: RBF kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    D = Xmu.shape[1]
    lengthscale = kern.lengthscale
    if not kern.ard:
        lengthscale = tf.zeros((D,), dtype=lengthscale.dtype) + lengthscale

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscale ** 2) + Xcov[0, :-1])  # NxDxD
    all_diffs = tf.transpose(feat.Z) - tf.expand_dims(Xmu[:-1], 2)  # NxDxM

    sqrt_det_L = tf.reduce_prod(lengthscale)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1))
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    exponent_mahalanobis = tf.linalg.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
    non_exponent_term = tf.linalg.matmul(Xcov[1, :-1], exponent_mahalanobis, transpose_a=True)
    non_exponent_term = tf.expand_dims(Xmu[1:], 2) + non_exponent_term  # NxDxM

    exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    return kern.variance * (determinants[:, None] *
                            exponent_mahalanobis)[:, None, :] * non_exponent_term


@dispatch.expectation.register((Gaussian, DiagonalGaussian), kernels.RBF, InducingPoints,
                               kernels.RBF, InducingPoints)
def _E(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: RBF kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other, but this is supported
    only if the Gaussian p is Diagonal (p.cov NxD) and Ka, Kb have disjoint active_dims
    in which case the joint expectations simplify into a product of expectations

    :return: NxMxM
    """
    if kern1.on_separate_dims(kern2) and isinstance(p, DiagonalGaussian):  # no joint expectations required
        eKxz1 = expectation(p, (kern1, feat1))
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if feat1 != feat2 or kern1 != kern2:
        raise NotImplementedError("The expectation over two kernels has only an "
                                  "analytical implementation if both kernels are equal.")

    kern = kern1
    feat = feat1

    # use only active dimensions
    Xcov = kern.slice_cov(tf.linalg.diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
    Z, Xmu = kern.slice(feat.Z, p.mu)

    N = Xmu.shape[0]
    D = Xmu.shape[1]

    squared_lengthscale = kern.lengthscale ** 2
    if not kern.ard:
        zero_lengthscale = tf.zeros((D,), dtype=squared_lengthscale.dtype)
        squared_lengthscale = squared_lengthscale + zero_lengthscale

    sqrt_det_L = tf.reduce_prod(0.5 * squared_lengthscale) ** 0.5
    C = tf.linalg.cholesky(0.5 * tf.linalg.diag(squared_lengthscale) + Xcov)  # NxDxD
    dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(C)), axis=1))  # N

    C_inv_mu = tf.linalg.triangular_solve(C, tf.expand_dims(Xmu, 2), lower=True)  # NxDx1
    C_inv_z = tf.linalg.triangular_solve(
        C,
        tf.tile(tf.expand_dims(tf.transpose(Z) / 2., 0), [N, 1, 1]),
        lower=True)  # NxDxM
    mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), 1), 2)  # Nx1x1
    z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), 1)  # NxM
    zm_CC_inv_zn = tf.linalg.matmul(C_inv_z, C_inv_z, transpose_a=True)  # NxMxM
    two_z_CC_inv_mu = 2 * tf.linalg.matmul(C_inv_z, C_inv_mu, transpose_a=True)[:, :, 0]  # NxM
    # NxMxM
    exponent_mahalanobis = mu_CC_inv_mu + tf.expand_dims(z_CC_inv_z, 1) + \
        tf.expand_dims(z_CC_inv_z, 2) + 2 * zm_CC_inv_zn - \
        tf.expand_dims(two_z_CC_inv_mu, 2) - tf.expand_dims(two_z_CC_inv_mu, 1)
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxMxM

    # Compute sqrt(self(Z)) explicitly to prevent automatic gradient from
    # being NaN sometimes, see pull request #615
    kernel_sqrt = tf.exp(-0.25 * kern.scaled_square_dist(Z, None))
    return kern.variance ** 2 * kernel_sqrt * tf.reshape(dets, [N, 1, 1]) * exponent_mahalanobis
