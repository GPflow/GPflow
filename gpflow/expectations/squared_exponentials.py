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

from typing import Type, Union

import tensorflow as tf

from .. import kernels
from .. import mean_functions as mfn
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from ..utilities.ops import square_distance
from . import dispatch
from .expectations import expectation

NoneType: Type[None] = type(None)


@dispatch.expectation.register(Gaussian, kernels.SquaredExponential, NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N]",
)
def _expectation_gaussian_sqe(
    p: Gaussian, kernel: kernels.SquaredExponential, _: None, __: None, ___: None, nghp: None = None
) -> tf.Tensor:
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
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M]",
)
def _expectation_gaussian_sqe_inducingpoints(
    p: Gaussian,
    kernel: kernels.SquaredExponential,
    inducing_variable: InducingPoints,
    _: None,
    __: None,
    nghp: None = None,
) -> tf.Tensor:
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
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, D, M]",
)
def _expectation_gaussian__sqe_inducingpoints(
    p: Gaussian,
    mean: mfn.Identity,
    _: None,
    kernel: kernels.SquaredExponential,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
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
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, D, M]",
)
def _expectation_markov__sqe_inducingpoints(
    p: MarkovGaussian,
    mean: mfn.Identity,
    _: None,
    kernel: kernels.SquaredExponential,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
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
@check_shapes(
    "p: [N, D]",
    "feat1: [M, D, P]",
    "feat2: [M, D, P]",
    "return: [N, M, M]",
)
def _expectation_gaussian_sqe_inducingpoints__sqe_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    kern1: kernels.SquaredExponential,
    feat1: InducingPoints,
    kern2: kernels.SquaredExponential,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: RBF kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other, but this is supported
    only if the Gaussian p is Diagonal (p.cov NxD) and Ka, Kb have disjoint active_dims
    in which case the joint expectations simplify into a product of expectations

    :return: NxMxM
    """
    if kern1.on_separate_dims(kern2) and isinstance(
        p, DiagonalGaussian
    ):  # no joint expectations required
        eKxz1 = expectation(p, (kern1, feat1))
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if feat1 != feat2 or kern1 != kern2:
        raise NotImplementedError(
            "The expectation over two kernels has only an "
            "analytical implementation if both kernels are equal."
        )

    kernel = kern1
    inducing_variable = feat1

    # use only active dimensions
    Xcov = kernel.slice_cov(tf.linalg.diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
    Z, Xmu = kernel.slice(inducing_variable.Z, p.mu)

    N = tf.shape(Xmu)[0]
    D = tf.shape(Xmu)[1]

    squared_lengthscales = kernel.lengthscales ** 2
    if not kernel.ard:
        zero_lengthscales = tf.zeros((D,), dtype=squared_lengthscales.dtype)
        squared_lengthscales = squared_lengthscales + zero_lengthscales

    sqrt_det_L = tf.reduce_prod(0.5 * squared_lengthscales) ** 0.5
    C = tf.linalg.cholesky(0.5 * tf.linalg.diag(squared_lengthscales) + Xcov)  # NxDxD
    dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.math.log(tf.linalg.diag_part(C)), axis=1))  # N

    C_inv_mu = tf.linalg.triangular_solve(C, tf.expand_dims(Xmu, 2), lower=True)  # NxDx1
    C_inv_z = tf.linalg.triangular_solve(
        C, tf.tile(tf.expand_dims(0.5 * tf.transpose(Z), 0), [N, 1, 1]), lower=True
    )  # NxDxM
    mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), 1), 2)  # Nx1x1
    z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), 1)  # NxM
    zm_CC_inv_zn = tf.linalg.matmul(C_inv_z, C_inv_z, transpose_a=True)  # NxMxM
    two_z_CC_inv_mu = 2 * tf.linalg.matmul(C_inv_z, C_inv_mu, transpose_a=True)[:, :, 0]  # NxM
    # NxMxM
    exponent_mahalanobis = (
        mu_CC_inv_mu
        + tf.expand_dims(z_CC_inv_z, 1)
        + tf.expand_dims(z_CC_inv_z, 2)
        + 2 * zm_CC_inv_zn
        - tf.expand_dims(two_z_CC_inv_mu, 2)
        - tf.expand_dims(two_z_CC_inv_mu, 1)
    )
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxMxM

    # Compute sqrt(self(Z)) explicitly to prevent automatic gradient from
    # being NaN sometimes, see pull request #615
    kernel_sqrt = tf.exp(-0.25 * square_distance(Z / kernel.lengthscales, None))
    return kernel.variance ** 2 * kernel_sqrt * tf.reshape(dets, [N, 1, 1]) * exponent_mahalanobis
