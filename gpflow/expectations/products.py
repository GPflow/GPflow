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

from functools import reduce
from typing import Type

import tensorflow as tf

from .. import kernels
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingPoints
from ..probability_distributions import DiagonalGaussian
from . import dispatch
from .expectations import expectation

NoneType: Type[None] = type(None)


@dispatch.expectation.register(DiagonalGaussian, kernels.Product, NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N]",
)
def _expectation_diagonal_product(
    p: DiagonalGaussian, kernel: kernels.Product, _: None, __: None, ___: None, nghp: None = None
) -> tf.Tensor:
    r"""
    Compute the expectation:
    <\HadamardProd_i diag(Ki_{X[:, active_dims_i], X[:, active_dims_i]})>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: N
    """
    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [expectation(p, k, nghp=nghp) for k in kernel.kernels]
    return reduce(tf.multiply, exps)


@dispatch.expectation.register(
    DiagonalGaussian, kernels.Product, InducingPoints, NoneType, NoneType
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M]",
)
def _expectation_diagonal_product_inducingpoints(
    p: DiagonalGaussian,
    kernel: kernels.Product,
    inducing_variable: InducingPoints,
    __: None,
    ___: None,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    <\HadamardProd_i Ki_{X[:, active_dims_i], Z[:, active_dims_i]}>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxM
    """
    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [expectation(p, (k, inducing_variable), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.multiply, exps)


@dispatch.expectation.register(
    DiagonalGaussian, kernels.Product, InducingPoints, kernels.Product, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M, D, P]",
    "feat2: [M, D, P]",
    "return: [N, M, M]",
)
def _expectation_diagonal_product_inducingpoints__product_inducingpoints(
    p: DiagonalGaussian,
    kern1: kernels.Product,
    feat1: InducingPoints,
    kern2: kernels.Product,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    expectation[n] = < prodK_{Z, x_n} prodK_{x_n, Z} >_p(x_n)
                   = < (\HadamardProd_i Ki_{Z[:, active_dims_i], x[n, active_dims_i]})  <-- Mx1
               1xM -->  (\HadamardProd_j Kj_{x[n, active_dims_j], Z[:, active_dims_j]}) >_p(x_n)  (MxM)

        - \HadamardProd_i Ki_{.,.}, \HadamardProd_j Kj_{.,.} :: Product kernels
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxMxM
    """
    if feat1 != feat2:
        raise NotImplementedError("Different inducing variables are not supported.")
    if kern1 != kern2:
        raise NotImplementedError(
            "Calculating the expectation over two " "different Product kernels is not supported."
        )

    kernel = kern1
    inducing_variable = feat1

    if not kernel.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions."
        )  # pragma: no cover

    exps = [
        expectation(p, (k, inducing_variable), (k, inducing_variable), nghp=nghp)
        for k in kernel.kernels
    ]
    return reduce(tf.multiply, exps)
