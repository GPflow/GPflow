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


@dispatch.expectation.register((Gaussian, DiagonalGaussian),
                               kernels.SquaredExponential,
                               InducingPoints,
                               kernels.SquaredExponential,
                               InducingPoints)
def _E(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
            - Ka_{.,.}, Kb_{.,.} :: RBF kernels

    :return: NxM1xM2
    """
    if kern1.on_separate_dims(kern2) and isinstance(p, DiagonalGaussian):
        eKxz1 = expectation(p, (kern1, feat1))  # No joint expectations required
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if kern1.on_separate_dims(kern2):
        raise NotImplementedError("The expectation over two kernels only has an "
                                  "analytical implementation if both kernels have "
                                  "the same active features.")

    is_same_kern = (kern1 == kern2)  # code branches by case for
    is_same_feat = (feat1 == feat2)  # computational efficiency

    mx = kern1.slice(p.mu)[0]
    if isinstance(p, DiagonalGaussian):
        Sxx = kern1.slice_cov(tf.linalg.diag(p.cov))
    else:
        Sxx = kern1.slice_cov(p.cov)

    N = tf.shape(mx)[0]  # num. random inputs $x$
    D = tf.shape(mx)[1]  # dimensionality of $x$

    # First Gaussian kernel $k1(x, z) = exp(-0.5*(x - z) V1^{-1} (x - z))$
    V1 = kern1.lengthscales ** 2  # D|1
    z1 = kern1.slice(feat1.Z)[0]  # M1xD
    iV1_z1 = (1/V1) * z1

    # Second Gaussian kernel $k2(x, z) = exp(-0.5*(x - z) V2^{-1} (x - z))$
    V2 = V1 if is_same_kern else kern2.lengthscales ** 2  # D|1
    z2 = z1 if is_same_feat else kern2.slice(feat2.Z)[0]  # M2xD
    iV2_z2 = iV1_z1 if (is_same_kern and is_same_feat) else (1/V2) * z2

    # Product of Gaussian kernels is another Gaussian kernel $k = k1 * k2$
    V = 0.5 * V1 if is_same_kern else (V1 * V2)/(V1 + V2)  # D|1
    if not (kern1.ard or kern2.ard):
            V = tf.fill((D,), V)  # D

    # Product of Gaussians is an unnormalized Gaussian; compute determinant of
    # this new Gaussian (and the Gaussian kernel) in order to normalize
    S = Sxx + tf.linalg.diag(V)
    L = tf.linalg.cholesky(S)  # NxDxD
    half_logdet_L = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(L)), axis=1)
    sqrt_det_iL = tf.exp(-half_logdet_L)
    sqrt_det_L = tf.sqrt(tf.reduce_prod(V))
    determinant = sqrt_det_L * sqrt_det_iL  # N

    # Solve for linear systems involving $S = LL^{T}$ where $S$
    # is the covariance of an (unnormalized) Gaussian distribution
    iL_mu = tf.linalg.triangular_solve(L,
                                       tf.expand_dims(mx, 2),
                                       lower=True)  # NxDx1

    V_iV1_z1 = tf.expand_dims(tf.transpose(V * iV1_z1), 0)
    iL_z1 = tf.linalg.triangular_solve(L,
                                       tf.tile(V_iV1_z1, [N, 1, 1]),
                                       lower=True)  # NxDxM1

    z1_iS_z1 = tf.reduce_sum(tf.square(iL_z1), axis=1)  # NxM1
    z1_iS_mu = tf.squeeze(tf.linalg.matmul(iL_z1, iL_mu, transpose_a=True), 2)  # NxM1
    if is_same_kern and is_same_feat:
        iL_z2 = iL_z1
        z2_iS_z2 = z1_iS_z1
        z2_iS_mu = z1_iS_mu
    else:
        V_iV2_z2 = tf.expand_dims(tf.transpose(V * iV2_z2), 0)
        iL_z2 = tf.linalg.triangular_solve(L,
                                           tf.tile(V_iV2_z2, [N, 1, 1]),
                                           lower=True)  # NxDxM2

        z2_iS_z2 = tf.reduce_sum(tf.square(iL_z2), 1)  # NxM2
        z2_iS_mu = tf.squeeze(tf.matmul(iL_z2, iL_mu, transpose_a=True), 2)  # NxM2

    z1_iS_z2 = tf.linalg.matmul(iL_z1, iL_z2, transpose_a=True)  # NxM1xM2
    mu_iS_mu = tf.expand_dims(tf.reduce_sum(tf.square(iL_mu), 1), 2)  # Nx1x1

    # Gram matrix from Gaussian integral of Gaussian kernel $k = k1 * k2$
    exp_mahalanobis = tf.exp(-0.5 * (mu_iS_mu + 2 * z1_iS_z2
        + tf.expand_dims(z1_iS_z1 - 2 * z1_iS_mu, axis=-1)
        + tf.expand_dims(z2_iS_z2 - 2 * z2_iS_mu, axis=-2)
    ))  # NxM1xM2

    # Part of $E_{p(x)}[k1(z1, x) k2(x, z2)]$ that is independent of $x$
    if is_same_kern:
        ampl2 = kern1.variance ** 2
        sq_iV = tf.math.rsqrt(V)
        if is_same_feat:
            matrix_term = ampl2 * tf.exp(-0.125 * square_distance(sq_iV * z1, None))
        else:
            matrix_term = ampl2 * tf.exp(-0.125 * square_distance(sq_iV * z1, sq_iV * z2))
    else:
        z1_iV1_z1 = tf.reduce_sum(z1 * iV1_z1, axis=-1)  # M1
        z2_iV2_z2 = tf.reduce_sum(z2 * iV2_z2, axis=-1)  # M2
        z1_iV1pV2_z1 = tf.reduce_sum(iV1_z1 * V * iV1_z1, axis=-1)
        z2_iV1pV2_z2 = tf.reduce_sum(iV2_z2 * V * iV2_z2, axis=-1)
        z1_iV1pV2_z2 = tf.matmul(iV1_z1, V * iV2_z2, transpose_b=True)  # M1xM2
        matrix_term = kern1.variance * kern2.variance * tf.exp(0.5 * (
                2 * z1_iV1pV2_z2  # implicit negative
                + tf.expand_dims(z1_iV1pV2_z1 - z1_iV1_z1, axis=-1)
                + tf.expand_dims(z2_iV1pV2_z2 - z2_iV2_z2, axis=-2)
        ))

    return tf.reshape(determinant, [N, 1, 1]) * matrix_term * exp_mahalanobis
