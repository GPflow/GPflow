# Copyright 2018 The GPflow Contributors. All Rights Reserved.
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
from typing import Optional, Tuple, Union, cast

import tensorflow as tf

from ..base import TensorType
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingVariables
from ..kernels import Kernel
from ..mean_functions import MeanFunction
from ..probability_distributions import (
    DiagonalGaussian,
    Gaussian,
    MarkovGaussian,
    ProbabilityDistribution,
)
from . import dispatch

ProbabilityDistributionLike = Union[ProbabilityDistribution, Tuple[TensorType, TensorType]]
"""
Either a prabability distribution, or a tuple of mean, covariance that is turned into an
appropriate Gaussian distribution, depending on the shape of the covariance.
"""

ExpectationObject = Union[Kernel, MeanFunction, None]
PackedExpectationObject = Union[ExpectationObject, Tuple[Kernel, InducingVariables]]


def expectation(
    p: ProbabilityDistributionLike,
    obj1: PackedExpectationObject,
    obj2: PackedExpectationObject = None,
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    Compute the expectation <obj1(x) obj2(x)>_p(x)
    Uses multiple-dispatch to select an analytical implementation,
    if one is available. If not, it falls back to quadrature.

    :type p: (mu, cov) tuple or a `ProbabilityDistribution` object
    :type obj1: kernel, mean function, (kernel, inducing_variable), or None
    :type obj2: kernel, mean function, (kernel, inducing_variable), or None
    :param int nghp: passed to `_quadrature_expectation` to set the number
                     of Gauss-Hermite points used: `num_gauss_hermite_points`
    :return: a 1-D, 2-D, or 3-D tensor containing the expectation

    Allowed combinations

    - Psi statistics:
        >>> eKdiag = expectation(p, kernel)  (N)  # Psi0
        >>> eKxz = expectation(p, (kernel, inducing_variable))  (NxM)  # Psi1
        >>> exKxz = expectation(p, identity_mean, (kernel, inducing_variable))  (NxDxM)
        >>> eKzxKxz = expectation(p, (kernel, inducing_variable), (kernel, inducing_variable))  (NxMxM)  # Psi2

    - kernels and mean functions:
        >>> eKzxMx = expectation(p, (kernel, inducing_variable), mean)  (NxMxQ)
        >>> eMxKxz = expectation(p, mean, (kernel, inducing_variable))  (NxQxM)

    - only mean functions:
        >>> eMx = expectation(p, mean)  (NxQ)
        >>> eM1x_M2x = expectation(p, mean1, mean2)  (NxQ1xQ2)
        .. note:: mean(x) is 1xQ (row vector)

    - different kernels. This occurs, for instance, when we are calculating Psi2 for Sum kernels:
        >>> eK1zxK2xz = expectation(p, (kern1, inducing_variable), (kern2, inducing_variable))  (NxMxM)
    """
    p, obj1, feat1, obj2, feat2 = _init_expectation(p, obj1, obj2)
    try:
        return dispatch.expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)
    except NotImplementedError:
        return dispatch.quadrature_expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)


def quadrature_expectation(
    p: ProbabilityDistributionLike,
    obj1: PackedExpectationObject,
    obj2: PackedExpectationObject = None,
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    Compute the expectation <obj1(x) obj2(x)>_p(x)
    Uses Gauss-Hermite quadrature for approximate integration.

    :type p: (mu, cov) tuple or a `ProbabilityDistribution` object
    :type obj1: kernel, mean function, (kernel, inducing_variable), or None
    :type obj2: kernel, mean function, (kernel, inducing_variable), or None
    :param int num_gauss_hermite_points: passed to `_quadrature_expectation` to set
                                         the number of Gauss-Hermite points used
    :return: a 1-D, 2-D, or 3-D tensor containing the expectation
    """
    print(f"2. p={p}, obj1={obj1}, obj2={obj2}")
    p, obj1, feat1, obj2, feat2 = _init_expectation(p, obj1, obj2)
    return dispatch.quadrature_expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)


@check_shapes(
    "return[0]: [N, D]",
    "return[2]: [M1, D, P]",
    "return[4]: [M2, D, P]",
)
def _init_expectation(
    p: ProbabilityDistributionLike, obj1: PackedExpectationObject, obj2: PackedExpectationObject
) -> Tuple[
    ProbabilityDistribution,
    ExpectationObject,
    Optional[InducingVariables],
    ExpectationObject,
    Optional[InducingVariables],
]:
    if isinstance(p, tuple):
        mu, cov = p
        classes = [DiagonalGaussian, Gaussian, MarkovGaussian]
        p = classes[cov.ndim - 2](*p)  # type: ignore[abstract]

    obj1, feat1 = obj1 if isinstance(obj1, tuple) else (obj1, None)
    obj2, feat2 = obj2 if isinstance(obj2, tuple) else (obj2, None)
    return (
        # type-ignore instead of cast, because it dependes on versions whether a cast is necessary.
        p,  # type: ignore[return-value]
        cast(ExpectationObject, obj1),
        cast(Optional[InducingVariables], feat1),
        cast(ExpectationObject, obj2),
        cast(Optional[InducingVariables], feat2),
    )
