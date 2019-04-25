import tensorflow as tf

from . import dispatch
from .. import kernels
from ..features import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian
from .expectations import expectation


@dispatch.expectation.register((Gaussian, DiagonalGaussian), kernels.RBF, InducingPoints, kernels.Linear, InducingPoints)
def _E(p, rbf_kern, feat1, lin_kern, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: RBF kernel
        - K_rbf_{.,.} :: Linear kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_rbf have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    if rbf_kern.on_separate_dims(lin_kern) and isinstance(p, DiagonalGaussian):  # no joint expectations required
        eKxz1 = expectation(p, (rbf_kern, feat1))
        eKxz2 = expectation(p, (lin_kern, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if feat1 != feat2:
        raise NotImplementedError("Features have to be the same for both kernels.")

    if rbf_kern.active_dims != lin_kern.active_dims:
        raise NotImplementedError("active_dims have to be the same for both kernels.")

    # use only active dimensions
    Xcov = rbf_kern.slice_cov(tf.linalg.diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
    Z, Xmu = rbf_kern.slice(feat1.Z, p.mu)

    N = Xmu.shape[0]
    D = Xmu.shape[1]

    def take_with_ard(value):
        if not rbf_kern.ard:
            return tf.zeros((D,), dtype=value.dtype) + value
        return value

    lin_kern_variances = take_with_ard(lin_kern.variance)
    rbf_kern_lengthscale = take_with_ard(rbf_kern.lengthscale)

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(rbf_kern_lengthscale ** 2) + Xcov)  # NxDxD

    Z_transpose = tf.transpose(Z)
    all_diffs = Z_transpose - tf.expand_dims(Xmu, 2)  # NxDxM
    exponent_mahalanobis = tf.linalg.triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
    exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    sqrt_det_L = tf.reduce_prod(rbf_kern_lengthscale)
    sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1))
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N
    eKxz_rbf = rbf_kern.variance * (determinants[:, None] * exponent_mahalanobis)  ## NxM <- End RBF eKxz code

    tiled_Z = tf.tile(tf.expand_dims(Z_transpose, 0), (N, 1, 1))  # NxDxM
    z_L_inv_Xcov = tf.linalg.matmul(tiled_Z, Xcov / rbf_kern_lengthscale[:, None] ** 2., transpose_a=True)  # NxMxD

    cross_eKzxKxz = tf.linalg.cholesky_solve(
        chol_L_plus_Xcov, (lin_kern_variances * rbf_kern_lengthscale ** 2.)[..., None] * tiled_Z)  # NxDxM

    cross_eKzxKxz = tf.linalg.matmul((z_L_inv_Xcov + Xmu[:, None, :]) * eKxz_rbf[..., None], cross_eKzxKxz)  # NxMxM
    return cross_eKzxKxz


@dispatch.expectation.register((Gaussian, DiagonalGaussian), kernels.Linear, InducingPoints, kernels.RBF, InducingPoints)
def _E(p, lin_kern, feat1, rbf_kern, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: Linear kernel
        - K_rbf_{.,.} :: RBF kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_rbf have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    return tf.linalg.adjoint(expectation(p, (rbf_kern, feat2), (lin_kern, feat1)))
