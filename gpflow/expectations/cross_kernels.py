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

from typing import Union, cast

import tensorflow as tf

from .. import kernels
from ..base import TensorType
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian, Gaussian
from . import dispatch
from .expectations import expectation


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian),
    kernels.SquaredExponential,
    InducingPoints,
    kernels.Linear,
    InducingPoints,
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M1, D, P]",
    "feat2: [M2, D, P]",
    "return: [N, M1, M2]",
)
def _expectation_gaussian_sqe_inducingpoints__linear_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    sqexp_kern: kernels.SquaredExponential,
    feat1: InducingPoints,
    lin_kern: kernels.Linear,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: SqExp kernel
        - K_sqexp_{.,.} :: Linear kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_sqexp have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    if sqexp_kern.on_separate_dims(lin_kern) and isinstance(
        p, DiagonalGaussian
    ):  # no joint expectations required
        eKxz1 = expectation(p, (sqexp_kern, feat1))
        eKxz2 = expectation(p, (lin_kern, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if feat1 != feat2:
        raise NotImplementedError("inducing_variables have to be the same for both kernels.")

    if sqexp_kern.active_dims != lin_kern.active_dims:
        raise NotImplementedError("active_dims have to be the same for both kernels.")

    # use only active dimensions
    Xcov = sqexp_kern.slice_cov(tf.linalg.diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
    Z, Xmu = sqexp_kern.slice(feat1.Z, p.mu)

    N = tf.shape(Xmu)[0]
    D = tf.shape(Xmu)[1]

    def take_with_ard(value: TensorType) -> TensorType:
        if not sqexp_kern.ard:
            return tf.zeros((D,), dtype=value.dtype) + value
        return value

    lin_kern_variances = take_with_ard(lin_kern.variance)
    sqexp_kern_lengthscales = take_with_ard(sqexp_kern.lengthscales)

    chol_L_plus_Xcov = tf.linalg.cholesky(
        tf.linalg.diag(sqexp_kern_lengthscales ** 2) + Xcov
    )  # NxDxD

    Z_transpose = tf.transpose(Z)
    all_diffs = Z_transpose - tf.expand_dims(Xmu, 2)  # NxDxM
    exponent_mahalanobis = tf.linalg.triangular_solve(
        chol_L_plus_Xcov, all_diffs, lower=True
    )  # NxDxM
    exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
    exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

    sqrt_det_L = tf.reduce_prod(sqexp_kern_lengthscales)
    sqrt_det_L_plus_Xcov = tf.exp(
        tf.reduce_sum(tf.math.log(tf.linalg.diag_part(chol_L_plus_Xcov)), axis=1)
    )
    determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N
    eKxz_sqexp = sqexp_kern.variance * (
        determinants[:, None] * exponent_mahalanobis
    )  ## NxM <- End RBF eKxz code

    tiled_Z = tf.tile(tf.expand_dims(Z_transpose, 0), (N, 1, 1))  # NxDxM
    z_L_inv_Xcov = tf.linalg.matmul(
        tiled_Z, Xcov / sqexp_kern_lengthscales[:, None] ** 2.0, transpose_a=True
    )  # NxMxD

    cross_eKzxKxz = tf.linalg.cholesky_solve(
        chol_L_plus_Xcov,
        cast(tf.Tensor, (lin_kern_variances * sqexp_kern_lengthscales ** 2.0))[..., None] * tiled_Z,
    )  # NxDxM

    cross_eKzxKxz = tf.linalg.matmul(
        (z_L_inv_Xcov + Xmu[:, None, :]) * eKxz_sqexp[..., None], cross_eKzxKxz
    )  # NxMxM
    return cross_eKzxKxz


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian),
    kernels.Linear,
    InducingPoints,
    kernels.SquaredExponential,
    InducingPoints,
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M1, D, P]",
    "feat2: [M2, D, P]",
    "return: [N, M1, M2]",
)
def _expectation_gaussian_linear_inducingpoints__sqe_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    lin_kern: kernels.Linear,
    feat1: InducingPoints,
    sqexp_kern: kernels.SquaredExponential,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: Linear kernel
        - K_sqexp_{.,.} :: sqexp kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_sqexp have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    return tf.linalg.adjoint(expectation(p, (sqexp_kern, feat2), (lin_kern, feat1)))
