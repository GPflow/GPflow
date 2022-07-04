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

from typing import Any, Callable, Optional, Type, Union, cast

import numpy as np
import tensorflow as tf

from .. import kernels
from .. import mean_functions as mfn
from ..base import TensorType
from ..covariances import Kuf
from ..experimental.check_shapes import check_shapes
from ..inducing_variables import InducingVariables
from ..probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian
from ..quadrature import mvnquad
from . import dispatch
from .expectations import ExpectationObject, PackedExpectationObject, quadrature_expectation

register = dispatch.quadrature_expectation.register


NoneType: Type[None] = type(None)
EllipsisType = Any


def get_eval_func(
    obj: ExpectationObject,
    inducing_variable: Optional[InducingVariables],
    slice: Union[slice, EllipsisType, None] = None,
) -> Callable[[TensorType], tf.Tensor]:
    """
    Return the function of interest (kernel or mean) for the expectation
    depending on the type of :obj: and whether any inducing are given
    """

    slice = ... if slice is None else slice
    if inducing_variable is not None:
        # kernel + inducing_variable combination
        if not isinstance(inducing_variable, InducingVariables) or not isinstance(
            obj, kernels.Kernel
        ):
            raise TypeError("If `inducing_variable` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(Kuf(inducing_variable, obj, x))[slice]
    elif isinstance(obj, mfn.MeanFunction):
        return lambda x: obj(x)[slice]  # type: ignore[misc]
    elif isinstance(obj, kernels.Kernel):
        return lambda x: obj(x, full_cov=False)  # type: ignore[call-arg, misc]

    raise NotImplementedError()


@dispatch.quadrature_expectation.register(
    (Gaussian, DiagonalGaussian),
    object,
    (InducingVariables, NoneType),
    object,
    (InducingVariables, NoneType),
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable1: [M1, D, P]",
    "inducing_variable2: [M2, D, P]",
    "return: [N, ...]",
)
def _quadrature_expectation_gaussian(
    p: Union[Gaussian, DiagonalGaussian],
    obj1: ExpectationObject,
    inducing_variable1: Optional[InducingVariables],
    obj2: ExpectationObject,
    inducing_variable2: Optional[InducingVariables],
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    General handling of quadrature expectations for Gaussians and DiagonalGaussians
    Fallback method for missing analytic expectations
    """
    nghp = 100 if nghp is None else nghp

    # logger.warning(
    #     "Quadrature is used to calculate the expectation. This means that "
    #     "an analytical implementations is not available for the given combination."
    # )

    if obj1 is None:
        raise NotImplementedError("First object cannot be None.")

    if not isinstance(p, DiagonalGaussian):
        cov = p.cov
    else:
        if (
            isinstance(obj1, kernels.Kernel)
            and isinstance(obj2, kernels.Kernel)
            and obj1.on_separate_dims(obj2)
        ):  # no joint expectations required
            eKxz1 = quadrature_expectation(
                p, cast(PackedExpectationObject, (obj1, inducing_variable1)), nghp=nghp
            )
            eKxz2 = quadrature_expectation(
                p, cast(PackedExpectationObject, (obj2, inducing_variable2)), nghp=nghp
            )
            return eKxz1[:, :, None] * eKxz2[:, None, :]
        cov = tf.linalg.diag(p.cov)

    if obj2 is None:

        def eval_func(x: TensorType) -> tf.Tensor:
            fn = get_eval_func(obj1, inducing_variable1)
            return fn(x)

    else:

        def eval_func(x: TensorType) -> tf.Tensor:
            fn1 = get_eval_func(obj1, inducing_variable1, np.s_[:, :, None])
            fn2 = get_eval_func(obj2, inducing_variable2, np.s_[:, None, :])
            return fn1(x) * fn2(x)

    return mvnquad(eval_func, p.mu, cov, nghp)


@dispatch.quadrature_expectation.register(
    MarkovGaussian, object, (InducingVariables, NoneType), object, (InducingVariables, NoneType)
)
@check_shapes(
    "p: [N, D]",
    "inducing_variable1: [M1, D, P]",
    "inducing_variable2: [M2, D, P]",
    "return: [N, ...]",
)
def _quadrature_expectation_markov(
    p: MarkovGaussian,
    obj1: ExpectationObject,
    inducing_variable1: Optional[InducingVariables],
    obj2: ExpectationObject,
    inducing_variable2: Optional[InducingVariables],
    nghp: Optional[int] = None,
) -> tf.Tensor:
    """
    Handling of quadrature expectations for Markov Gaussians (useful for time series)
    Fallback method for missing analytic expectations wrt Markov Gaussians
    Nota Bene: obj1 is always associated with x_n, whereas obj2 always with x_{n+1}
               if one requires e.g. <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1}), compute the
               transpose and then transpose the result of the expectation
    """
    nghp = 40 if nghp is None else nghp

    # logger.warning(
    #     "Quadrature is used to calculate the expectation. This means that "
    #     "an analytical implementations is not available for the given combination."
    # )

    if obj2 is None:

        def eval_func(x: TensorType) -> tf.Tensor:
            return get_eval_func(obj1, inducing_variable1)(x)

        mu, cov = p.mu[:-1], p.cov[0, :-1]  # cross covariances are not needed
    elif obj1 is None:

        def eval_func(x: TensorType) -> tf.Tensor:
            return get_eval_func(obj2, inducing_variable2)(x)

        mu, cov = p.mu[1:], p.cov[0, 1:]  # cross covariances are not needed
    else:

        def eval_func(x: TensorType) -> tf.Tensor:
            x1 = tf.split(x, 2, 1)[0]
            x2 = tf.split(x, 2, 1)[1]
            res1 = get_eval_func(obj1, inducing_variable1, np.s_[:, :, None])(x1)
            res2 = get_eval_func(obj2, inducing_variable2, np.s_[:, None, :])(x2)
            return res1 * res2

        mu = tf.concat((p.mu[:-1, :], p.mu[1:, :]), 1)  # Nx2D
        cov_top = tf.concat((p.cov[0, :-1, :, :], p.cov[1, :-1, :, :]), 2)  # NxDx2D
        cov_bottom = tf.concat((tf.linalg.adjoint(p.cov[1, :-1, :, :]), p.cov[0, 1:, :, :]), 2)
        cov = tf.concat((cov_top, cov_bottom), 1)  # Nx2Dx2D

    return mvnquad(eval_func, mu, cov, nghp)
