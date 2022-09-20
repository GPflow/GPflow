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

import itertools
from functools import reduce
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


@dispatch.expectation.register(Gaussian, kernels.Sum, NoneType, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "return: [N]",
)
def _expectation_gaussian_sum(
    p: Gaussian, kernel: kernels.Sum, _: None, __: None, ___: None, nghp: None = None
) -> tf.Tensor:
    r"""
    Compute the expectation:
    <\Sum_i diag(Ki_{X, X})>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: N
    """
    exps = [expectation(p, k, nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@dispatch.expectation.register(Gaussian, kernels.Sum, InducingPoints, NoneType, NoneType)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M]",
)
def _expectation_gaussian_sum_inducingpoints(
    p: Gaussian,
    kernel: kernels.Sum,
    inducing_variable: InducingPoints,
    _: None,
    __: None,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    <\Sum_i Ki_{X, Z}>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxM
    """
    exps = [expectation(p, (k, inducing_variable), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@dispatch.expectation.register(
    Gaussian, (mfn.Linear, mfn.Identity, mfn.Constant), NoneType, kernels.Sum, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, Q, M]",
)
def _expectation_gaussian_linear__sum_inducingpoints(
    p: Gaussian,
    mean: Union[mfn.Linear, mfn.Identity, mfn.Constant],
    _: None,
    kernel: kernels.Sum,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    expectation[n] = <m(x_n)^T (\Sum_i Ki_{x_n, Z})>_p(x_n)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxQxM
    """
    exps = [expectation(p, mean, (k, inducing_variable), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@dispatch.expectation.register(MarkovGaussian, mfn.Identity, NoneType, kernels.Sum, InducingPoints)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, D, M]",
)
def _expectation_markov__sum_inducingpoints(
    p: MarkovGaussian,
    mean: mfn.Identity,
    _: None,
    kernel: kernels.Sum,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    expectation[n] = <x_{n+1} (\Sum_i Ki_{x_n, Z})>_p(x_{n:n+1})
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxDxM
    """
    exps = [expectation(p, mean, (k, inducing_variable), nghp=nghp) for k in kernel.kernels]
    return reduce(tf.add, exps)


@dispatch.expectation.register(
    (Gaussian, DiagonalGaussian), kernels.Sum, InducingPoints, kernels.Sum, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "feat1: [M1, D, P]",
    "feat2: [M2, D, P]",
    "return: [N, M1, M2]",
)
def _expectation_gaussian_sum_inducingpoints__sum_inducingpoints(
    p: Union[Gaussian, DiagonalGaussian],
    kern1: kernels.Sum,
    feat1: InducingPoints,
    kern2: kernels.Sum,
    feat2: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    r"""
    Compute the expectation:
    expectation[n] = <(\Sum_i K1_i_{Z1, x_n}) (\Sum_j K2_j_{x_n, Z2})>_p(x_n)
        - \Sum_i K1_i_{.,.}, \Sum_j K2_j_{.,.} :: Sum kernels

    :return: NxM1xM2
    """
    crossexps = []

    if kern1 == kern2 and feat1 == feat2:  # avoid duplicate computation by using transposes
        for i, k1 in enumerate(kern1.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k1, feat1), nghp=nghp))

            for k2 in kern1.kernels[:i]:
                eKK = expectation(p, (k1, feat1), (k2, feat2), nghp=nghp)
                eKK += tf.linalg.adjoint(eKK)
                crossexps.append(eKK)
    else:
        for k1, k2 in itertools.product(kern1.kernels, kern2.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k2, feat2), nghp=nghp))

    return reduce(tf.add, crossexps)
