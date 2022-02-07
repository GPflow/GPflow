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

from typing import Union

import tensorflow as tf

from ...inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
)
from ...kernels import (
    IndependentLatent,
    LinearCoregionalization,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)
from ..dispatch import Kuu


@Kuu.register(InducingPoints, MultioutputKernel)
def Kuu_generic(
    inducing_variable: InducingPoints, kernel: MultioutputKernel, *, jitter: float = 0.0
) -> tf.Tensor:
    Kmm = kernel(inducing_variable.Z, full_cov=True, full_output_cov=True)  # [M, P, M, P]
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), tf.shape(Kmm))
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def Kuu_shared_shared(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, (SeparateIndependent, IndependentLatent))
def Kuu_fallback_shared(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: Union[SeparateIndependent, IndependentLatent],
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = tf.stack(
        [Kuu(inducing_variable.inducing_variable, k) for k in kernel.kernels], axis=0
    )  # [L, M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables, SharedIndependent)
def Kuu_fallback_separate_shared(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmm = tf.stack(
        [Kuu(f, kernel.kernel) for f in inducing_variable.inducing_variable_list], axis=0
    )  # [L, M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(
    FallbackSeparateIndependentInducingVariables, (SeparateIndependent, LinearCoregionalization)
)
def Kuu_fallbace_separate(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: Union[SeparateIndependent, LinearCoregionalization],
    *,
    jitter: float = 0.0,
) -> tf.Tensor:
    Kmms = [Kuu(f, k) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(inducing_variable.num_inducing, dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat
