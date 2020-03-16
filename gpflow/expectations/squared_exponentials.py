import tensorflow as tf

from .. import kernels
from .. import mean_functions as mfn
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from ..utilities.ops import square_distance
from . import dispatch
from .expectations import expectation

NoneType = type(None)


@dispatch.expectation.register(Gaussian, kernels.SquaredExponential, NoneType, NoneType, NoneType)
def _E(p, kernel, _, __, ___, nghp=None):
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: RBF kernel

    :return: N
    """
    return kernel(p.mu, full_cov=False)


@dispatch.expectation.register(
    Gaussian, kernels.SquaredExponential, InducingPoints, NoneType, NoneType
)
def _E(p, kernel, inducing_variable, _, __, nghp=None):
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: RBF kernel

    :return: NxM
    """
    # use only active dimensions
    Xcov = kernel.slice_cov(p.cov)
    Z, Xmu = kernel.slice(inducing_variable.Z, p.mu)
    D = tf.shape(Xmu)[1]

    lengthscales = kernel.lengthscales
    if not kernel.ard:
        lengthscales = tf.zeros((D,), dtype=lengthscales.dtype) + kernel.lengthscales

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscales ** 2) + Xcov)  # NxDxD

    all_diffs = tf.transpose(Z) - tf.expand_dims(Xmu, 2)  # NxDxM
    exponent_mahalanobis = tf.linalg.triangular_solve(
        chol_L_plus_Xcov, all_diffs, lower=True
    )  # NxDxM
    exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    sqrt_det_L = tf.reduce_prod(lengthscales)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1)
    )
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    return kernel.variance * (determinants[:, None] * exponent_mahalanobis)


@dispatch.expectation.register(
    Gaussian, mfn.Identity, NoneType, kernels.SquaredExponential, InducingPoints
)
def _E(p, mean, _, kernel, inducing_variable, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_n K_{x_n, Z}>_p(x_n)
        - K_{.,.} :: RBF kernel

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    D = tf.shape(Xmu)[1]

    lengthscales = kernel.lengthscales
    if not kernel.ard:
        lengthscales = tf.zeros((D,), dtype=lengthscales.dtype) + lengthscales

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscales ** 2) + Xcov)  # NxDxD
    all_diffs = tf.transpose(inducing_variable.Z) - tf.expand_dims(Xmu, 2)  # NxDxM

    sqrt_det_L = tf.reduce_prod(lengthscales)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1)
    )
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    exponent_mahalanobis = tf.linalg.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
    non_exponent_term = tf.linalg.matmul(Xcov, exponent_mahalanobis, transpose_a=True)
    non_exponent_term = tf.expand_dims(Xmu, 2) + non_exponent_term  # NxDxM

    exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    return (
        kernel.variance
        * (determinants[:, None] * exponent_mahalanobis)[:, None, :]
        * non_exponent_term
    )


@dispatch.expectation.register(
    MarkovGaussian, mfn.Identity, NoneType, kernels.SquaredExponential, InducingPoints
)
def _E(p, mean, _, kernel, inducing_variable, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1})
        - K_{.,.} :: RBF kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    D = tf.shape(Xmu)[1]
    lengthscales = kernel.lengthscales
    if not kernel.ard:
        lengthscales = tf.zeros((D,), dtype=lengthscales.dtype) + lengthscales

    chol_L_plus_Xcov = tf.linalg.cholesky(tf.linalg.diag(lengthscales ** 2) + Xcov[0, :-1])  # NxDxD
    all_diffs = tf.transpose(inducing_variable.Z) - tf.expand_dims(Xmu[:-1], 2)  # NxDxM

    sqrt_det_L = tf.reduce_prod(lengthscales)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1)
    )
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

    exponent_mahalanobis = tf.linalg.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
    non_exponent_term = tf.linalg.matmul(Xcov[1, :-1], exponent_mahalanobis, transpose_a=True)
    non_exponent_term = tf.expand_dims(Xmu[1:], 2) + non_exponent_term  # NxDxM

    exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    return (
        kernel.variance
        * (determinants[:, None] * exponent_mahalanobis)[:, None, :]
        * non_exponent_term
    )


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian),
    kernels.SquaredExponential,
    InducingPoints,
    kernels.SquaredExponential,
    InducingPoints,
)
def _E(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: RBF kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other.

    :return: [N, dim(Z1), dim(Z2)]
    """
    if kern1.on_separate_dims(kern2) and isinstance(
        p, DiagonalGaussian
    ):  # no joint expectations required
        eKxz1 = expectation(p, (kern1, feat1))
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    Ka, Kb = kern1, kern2

    iv1, iv2 = feat1, feat2

    # use only active dimensions
    Xcov = Ka.slice_cov(tf.linalg.diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
    Z1, Xmu = Ka.slice(iv1.Z, p.mu)

    N = tf.shape(Xmu)[0]
    D = tf.shape(Xmu)[1]

    def get_squared_lengthscales(kernel):
        squared_lengthscales = kernel.lengthscales ** 2
        if not kernel.ard:
            # expand scalar lengthscale
            squared_lengthscales = (
                tf.ones((D,), dtype=squared_lengthscales.dtype) * squared_lengthscales
            )
        return squared_lengthscales

    if Ka == Kb:
        La = get_squared_lengthscales(Ka)
        Lb = La
        half_mean_L = La * 0.5  # average length scale
    else:
        La = get_squared_lengthscales(Ka)
        Lb = get_squared_lengthscales(Kb)
        half_mean_L = (La * Lb) / (La + Lb)  # average length scale

    sqrt_det_L = tf.reduce_prod(half_mean_L) ** 0.5
    C = tf.linalg.cholesky(tf.linalg.diag(half_mean_L) + Xcov)  # [N, D, D]
    dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(C)), axis=1))  # [N]

    # for Mahalanobis computation we need Zᵀ (CCᵀ)⁻¹ Z  as well as C⁻¹ Z
    # with Z = Z₁, Z₂  for two squared exponential kernels
    def get_cholesky_solve_terms(Z, C=C):
        C_inv_z = tf.linalg.triangular_solve(
            C, tf.tile(tf.expand_dims(tf.transpose(Z), axis=0), [N, 1, 1]), lower=True
        )  # [N, D, M]
        z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), axis=1)  # [N, M]

        return C_inv_z, z_CC_inv_z

    C_inv_mu = tf.linalg.triangular_solve(C, tf.expand_dims(Xmu, axis=2), lower=True)  # [N, D, 1]
    mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), axis=1), axis=2)  # [N, 1, 1]

    C_inv_z1, z1_CC_inv_z1 = get_cholesky_solve_terms(Z1 / La * half_mean_L)
    z1_CC_inv_mu = 2 * tf.matmul(C_inv_z1, C_inv_mu, transpose_a=True)[:, :, 0]  # [N, M1]

    if iv1 == iv2 and Ka == Kb:
        # in this case Z2==Z1 so we can reuse the Z1 terms
        C_inv_z2, z2_CC_inv_z2 = C_inv_z1, z1_CC_inv_z1
        z2_CC_inv_mu = z1_CC_inv_mu  # [N, M]
        Z2 = Z1
    else:
        # compute terms related to Z2
        Z2, _ = Kb._slice(iv2.Z, p.mu)
        C_inv_z2, z2_CC_inv_z2 = get_cholesky_solve_terms(Z2 / Lb * half_mean_L)
        z2_CC_inv_mu = 2 * tf.matmul(C_inv_z2, C_inv_mu, transpose_a=True)[:, :, 0]  # [N, M2]

    z1_CC_inv_z2 = tf.matmul(C_inv_z1, C_inv_z2, transpose_a=True)  # [N, M1, M2]

    # expand dims for broadcasting
    # along M1
    z2_CC_inv_mu = tf.expand_dims(z2_CC_inv_mu, axis=1)  # [N, 1, M2]
    z2_CC_inv_z2 = tf.expand_dims(z2_CC_inv_z2, axis=1)

    # along M2
    z1_CC_inv_mu = tf.expand_dims(z1_CC_inv_mu, axis=2)  # [N, M1, 1]
    z1_CC_inv_z1 = tf.expand_dims(z1_CC_inv_z1, axis=2)

    # expanded version of ((Z1 + Z2)-mu) (CCT)-1 ((Z1 + Z2)-mu)
    mahalanobis = (
        mu_CC_inv_mu + z2_CC_inv_z2 + z1_CC_inv_z1 + 2 * z1_CC_inv_z2 - z1_CC_inv_mu - z2_CC_inv_mu
    )  # [N, M1, M2]

    exp_mahalanobis = tf.exp(-0.5 * mahalanobis)  # [N, M1, M2]

    if Z1 == Z2:
        # CAVEAT : Compute sqrt(self.K(Z)) explicitly
        # to prevent automatic gradient from
        # being NaN sometimes, see https://github.com/GPflow/GPflow/pull/615
        sqrt_exp_dist = tf.exp(-0.25 * Ka.scaled_square_dist(Z1, None))
    else:
        # Compute exp( -.5 (Z-Z')^top (L_1+L_2)^{-1} (Z-Z') )
        lengthscales_rms = tf.sqrt(La + Lb)
        Z1scaled = Z1 / lengthscales_rms
        Z1sqr = tf.reduce_sum(tf.square(Z1scaled), axis=1)
        Z2scaled = Z2 / lengthscales_rms
        Z2sqr = tf.reduce_sum(tf.square(Z2scaled), axis=1)
        dist = (
            -2 * tf.matmul(Z1scaled, Z2scaled, transpose_b=True)
            + tf.reshape(Z1sqr, (-1, 1))
            + tf.reshape(Z2sqr, (1, -1))
        )
        sqrt_exp_dist = tf.exp(-0.5 * dist)  # [M1, M2]

    return Ka.variance * Kb.variance * sqrt_exp_dist * tf.reshape(dets, [N, 1, 1]) * exp_mahalanobis
