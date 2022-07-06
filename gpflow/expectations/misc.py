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

from typing import Optional, Type, Union, cast

import tensorflow as tf

from .. import kernels
from .. import mean_functions as mfn
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingPoints, InducingVariables
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from . import dispatch
from .expectations import ExpectationObject, PackedExpectationObject, expectation

NoneType: Type[None] = type(None)

# ================ exKxz transpose and mean function handling =================


@dispatch.expectation.register(
    (Gaussian, MarkovGaussian), mfn.Identity, NoneType, kernels.Linear, InducingPoints
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, D, M]",
)
def _expectation_gaussian__linear_inducingpoints(
    p: Union[Gaussian, MarkovGaussian],
    mean: mfn.Identity,
    _: None,
    kernel: kernels.Linear,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <x_n K_{x_n, Z}>_p(x_n)
        - K_{.,} :: Linear kernel
    or the equivalent for MarkovGaussian

    :return: NxDxM
    """
    return tf.linalg.adjoint(expectation(p, (kernel, inducing_variable), mean))


@dispatch.expectation.register(
    (Gaussian, MarkovGaussian), kernels.Kernel, InducingVariables, mfn.MeanFunction, NoneType
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, M, Q]",
)
def _expectation_gaussian_kernel_inducingvariables__meanfunction(
    p: Union[Gaussian, MarkovGaussian],
    kernel: kernels.Kernel,
    inducing_variable: InducingVariables,
    mean: mfn.MeanFunction,
    _: None,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} m(x_n)>_p(x_n)
    or the equivalent for MarkovGaussian

    :return: NxMxQ
    """
    return tf.linalg.adjoint(expectation(p, mean, (kernel, inducing_variable), nghp=nghp))


@dispatch.expectation.register(Gaussian, mfn.Constant, NoneType, kernels.Kernel, InducingPoints)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, Q, M]",
)
def _expectation_gaussian_constant__kernel_inducingpoints(
    p: Gaussian,
    constant_mean: mfn.Constant,
    _: None,
    kernel: kernels.Kernel,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T K_{x_n, Z}>_p(x_n)
        - m(x_i) = c :: Constant function
        - K_{.,.}    :: Kernel function

    :return: NxQxM
    """
    c = constant_mean(p.mu)  # NxQ
    eKxz = expectation(p, (kernel, inducing_variable), nghp=nghp)  # NxM

    return c[..., None] * eKxz[:, None, :]


@dispatch.expectation.register(Gaussian, mfn.Linear, NoneType, kernels.Kernel, InducingPoints)
@check_shapes(
    "p: [N, D]",
    "inducing_variable: [M, D, P]",
    "return: [N, Q, M]",
)
def _expectation_gaussian_linear__kernel_inducingpoints(
    p: Gaussian,
    linear_mean: mfn.Linear,
    _: None,
    kernel: kernels.Kernel,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T K_{x_n, Z}>_p(x_n)
        - m(x_i) = A x_i + b :: Linear mean function
        - K_{.,.}            :: Kernel function

    :return: NxQxM
    """
    N = tf.shape(p.mu)[0]
    D = tf.shape(p.mu)[1]
    exKxz = expectation(p, mfn.Identity(D), (kernel, inducing_variable), nghp=nghp)
    eKxz = expectation(p, (kernel, inducing_variable), nghp=nghp)
    eAxKxz = tf.linalg.matmul(
        tf.tile(linear_mean.A[None, :, :], (N, 1, 1)), exKxz, transpose_a=True
    )
    ebKxz = linear_mean.b[None, :, None] * eKxz[:, None, :]
    return eAxKxz + ebKxz


@dispatch.expectation.register(Gaussian, mfn.Identity, NoneType, kernels.Kernel, InducingPoints)
def _expectation_gaussian__kernel_inducingpoints(
    p: Gaussian,
    identity_mean: mfn.Identity,
    _: None,
    kernel: kernels.Kernel,
    inducing_variable: InducingPoints,
    nghp: None = None,
) -> tf.Tensor:
    """
    This prevents infinite recursion for kernels that don't have specific
    implementations of _expectation(p, identity_mean, None, kernel, inducing_variable).
    Recursion can arise because Identity is a subclass of Linear mean function
    so _expectation(p, linear_mean, none, kernel, inducing_variable) would call itself.
    More specific signatures (e.g. (p, identity_mean, None, RBF, inducing_variable)) will
    be found and used whenever available
    """
    raise NotImplementedError


# ============== Conversion to Gaussian from Diagonal or Markov ===============
# Catching missing DiagonalGaussian implementations by converting to full Gaussian:


@dispatch.expectation.register(
    DiagonalGaussian, object, (InducingVariables, NoneType), object, (InducingVariables, NoneType)
)
def _expectation_diagonal_generic(
    p: DiagonalGaussian,
    obj1: ExpectationObject,
    feat1: Optional[InducingVariables],
    obj2: ExpectationObject,
    feat2: Optional[InducingVariables],
    nghp: None = None,
) -> tf.Tensor:
    gaussian = Gaussian(p.mu, tf.linalg.diag(p.cov))
    return expectation(
        gaussian,
        cast(PackedExpectationObject, (obj1, feat1)),
        cast(PackedExpectationObject, (obj2, feat2)),
        nghp=nghp,
    )


# Catching missing MarkovGaussian implementations by converting to Gaussian (when indifferent):


@dispatch.expectation.register(
    MarkovGaussian, object, (InducingVariables, NoneType), object, (InducingVariables, NoneType)
)
def _expectation_markov_generic(
    p: MarkovGaussian,
    obj1: ExpectationObject,
    feat1: Optional[InducingVariables],
    obj2: ExpectationObject,
    feat2: Optional[InducingVariables],
    nghp: None = None,
) -> tf.Tensor:
    """
    Nota Bene: if only one object is passed, obj1 is
    associated with x_n, whereas obj2 with x_{n+1}

    """
    if obj2 is None:
        gaussian = Gaussian(p.mu[:-1], p.cov[0, :-1])
        return expectation(gaussian, cast(PackedExpectationObject, (obj1, feat1)), nghp=nghp)
    elif obj1 is None:
        gaussian = Gaussian(p.mu[1:], p.cov[0, 1:])
        return expectation(gaussian, cast(PackedExpectationObject, (obj2, feat2)), nghp=nghp)
    else:
        return expectation(
            p,
            cast(PackedExpectationObject, (obj1, feat1)),
            cast(PackedExpectationObject, (obj2, feat2)),
            nghp=nghp,
        )
