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
from . import dispatch
from .expectations import expectation

NoneType: Type[None] = type(None)


@dispatch.expectation.register(Gaussian, kernels.Linear, NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N]",
)
def _expectation_gaussian_linear(
    p: Gaussian, kernel: kernels.Linear, _: None, __: None, ___: None, nghp: None = None
) -> tf.Tensor:
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: Linear kernel

    :return: N
    """
    # use only active dimensions
    Xmu, _ = kernel.slice(p.mu, None)
    Xcov = kernel.slice_cov(p.cov)

    return tf.reduce_sum(kernel.variance * (tf.linalg.diag_part(Xcov) + Xmu ** 2), 1)


@dispatch.expectation.register(Gaussian, kernels.Linear, InducingPoints, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M]",
)
def _expectation_gaussian_linear_inducingpoints(
    p: Gaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    _: None,
    __: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: Linear kernel

    :return: NxM
    """
    # use only active dimensions
    Z, Xmu = kernel.slice(inducing_variable.Z, p.mu)

    return tf.linalg.matmul(Xmu, Z * kernel.variance, transpose_b=True)


@dispatch.expectation.register(Gaussian, kernels.Linear, InducingPoints, mfn.Identity, NoneType)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M, D]",
)
def _expectation_gaussian_linear_inducingpoints__identity(
    p: Gaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    mean: mfn.Identity,
    _: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_n^T>_p(x_n)
        - K_{.,.} :: Linear kernel

    :return: NxMxD
    """
    Xmu, Xcov = p.mu, p.cov

    N = tf.shape(Xmu)[0]
    var_Z = kernel.variance * inducing_variable.Z  # MxD
    tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
    return tf.linalg.matmul(tiled_Z, Xcov + (Xmu[..., None] * Xmu[:, None, :]))


@dispatch.expectation.register(
    MarkovGaussian, kernels.Linear, InducingPoints, mfn.Identity, NoneType
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M, D]",
)
def _expectation_markov_linear_inducingpoints__identity(
    p: MarkovGaussian,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    mean: mfn.Identity,
    _: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_{n+1}^T>_p(x_{n:n+1})
        - K_{.,.} :: Linear kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxMxD
    """
    Xmu, Xcov = p.mu, p.cov

    N = tf.shape(Xmu)[0] - 1
    var_Z = kernel.variance * inducing_variable.Z  # MxD
    tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
    eXX = Xcov[1, :-1] + (Xmu[:-1][..., None] * Xmu[1:][:, None, :])  # NxDxD
    return tf.linalg.matmul(tiled_Z, eXX)


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian), kernels.Linear, InducingPoints, kernels.Linear, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M, D, P]",
    "feat2: [M, D, P]",
    "return: [N, M, M]",
)
def _expectation_gaussian_linear_inducingpoints__linear_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    kern1: kernels.Linear,
    feat1: InducingPoints,
    kern2: kernels.Linear,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: Linear kernels
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

    if kern1 != kern2 or feat1 != feat2:
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
    var_Z = kernel.variance * Z
    tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
    XX = Xcov + tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2)  # NxDxD
    return tf.linalg.matmul(tf.linalg.matmul(tiled_Z, XX), tiled_Z, transpose_b=True)
