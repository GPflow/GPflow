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
def _Kuf(inducing_variable: InducingPoints, kernel: MultioutputKernel, Xnew: tf.Tensor):
    return kernel(inducing_variable.Z, Xnew, full_cov=True, full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentInducingVariables, SharedIndependent, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
):
    return Kuf(inducing_variable.inducing_variable, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentInducingVariables, SharedIndependent, object)
def _Kuf(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(f, kernel.kernel, Xnew) for f in inducing_variable.inducing_variable_list], axis=0
    )  # [L, M, N]


@Kuf.register(SharedIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, SeparateIndependent, object)
def _Kuf(
    inducing_variable: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    Kufs = [
        Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)
    ]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register(
    (FallbackSeparateIndependentInducingVariables, FallbackSharedIndependentInducingVariables),
    LinearCoregionalization,
    object,
)
def _Kuf(
    inducing_variable: Union[
        SeparateIndependentInducingVariables, SharedIndependentInducingVariables
    ],
    kernel: LinearCoregionalization,
    Xnew: tf.Tensor,
):
    kuf_impl = Kuf.dispatch(type(inducing_variable), SeparateIndependent, object)
    K = tf.transpose(kuf_impl(inducing_variable, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :, None, :]  # [M, L, N, P]


@Kuf.register(SharedIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(
    inducing_variable: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: tf.Tensor,
):
    return tf.stack(
        [Kuf(inducing_variable.inducing_variable, k, Xnew) for k in kernel.kernels], axis=0
    )  # [L, M, N]


@Kuf.register(SeparateIndependentInducingVariables, LinearCoregionalization, object)
def _Kuf(inducing_variable, kernel, Xnew):
    return tf.stack(
        [Kuf(f, k, Xnew) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)],
        axis=0,
    )  # [L, M, N]
