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

from typing import Callable, Union

import tensorflow as tf

from ...base import TensorType
from ...experimental.check_shapes import check_shapes
from ...inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import (
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from ..dispatch import Kuf


@Kuf.register(InducingPoints, MultioutputKernel, object)
@check_shapes(
    "inducing_variable: [M, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M, P, batch..., N, P]",
)
def Kuf_generic(
    inducing_variable: InducingPoints, kernel: MultioutputKernel, Xnew: TensorType
) -> tf.Tensor:
    return kernel(inducing_variable.Z, Xnew, full_cov=True, full_output_cov=True)


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [M, batch..., N]",
)
def Kuf_shared_shared(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
) -> tf.Tensor:
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_shared(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    return tf.stack(
        [Kuf(f, kernel.kernel, Xnew) for f in inducing_variable.inducing_variable_list], axis=0
    )


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_shared_separate(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_separate(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
) -> tf.Tensor:
    n_iv = len(inducing_variable.inducing_variable_list)
    n_k = len(kernel.kernels)
    assert (
        n_iv == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv} and {n_k}."

    Kufs = [
        Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)
    ]
    return tf.stack(Kufs, axis=0)


@check_shapes(
    "inducing_variable: [M, D, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def _fallback_Kuf(
    kuf_impl: Callable[
        [
            Union[SeparateIndependentInducingVariables, SharedIndependentInducingVariables],
            LinearCoregionalization,
            TensorType,
        ],
        tf.Tensor,
    ],
    inducing_variable: Union[
        SeparateIndependentInducingVariables, SharedIndependentInducingVariables
    ],
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    K = tf.transpose(kuf_impl(inducing_variable, kernel, Xnew), [1, 0, 2])
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]


@Kuf.register(
    FallbackSeparateIndependentInducingVariables,
    LinearCoregionalization,
    object,
)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def Kuf_fallback_separate_linear_coregionalization(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    kuf_impl = Kuf.dispatch_or_raise(
        SeparateIndependentInducingVariables, SeparateIndependent, object
    )
    return _fallback_Kuf(kuf_impl, inducing_variable, kernel, Xnew)


@Kuf.register(
    FallbackSharedIndependentInducingVariables,
    LinearCoregionalization,
    object,
)
@check_shapes(
    "inducing_variable: [M, D, P]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [M, L, batch..., N, P]",
)
def Kuf_fallback_shared_linear_coregionalization(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    kuf_impl = Kuf.dispatch_or_raise(
        SharedIndependentInducingVariables, SeparateIndependent, object
    )
    return _fallback_Kuf(kuf_impl, inducing_variable, kernel, Xnew)


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_shared_linear_coregionalization(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalization, object)
@check_shapes(
    "inducing_variable: [M, D, L]",
    "kernel.W: [P, L]",
    "Xnew: [batch..., N, D]",
    "return: [L, M, batch..., N]",
)
def Kuf_separate_linear_coregionalization(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: LinearCoregionalization,
    Xnew: TensorType,
) -> tf.Tensor:
    n_iv = len(inducing_variable.inducing_variable_list)
    n_k = len(kernel.kernels)
    assert (
        n_iv == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv} and {n_k}."

    return tf.stack(
        [Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)],
        axis=0,
    )
