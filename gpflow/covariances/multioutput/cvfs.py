# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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

from typing import Optional

import tensorflow as tf
from check_shapes import check_shapes

from gpflow.base import TensorType
from gpflow.covariances.dispatch import Cvf
from gpflow.inducing_variables import (
    InducingPoints,
    SeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from gpflow.kernels import MultioutputKernel, SeparateIndependent, SharedIndependent


@Cvf.register(InducingPoints, InducingPoints, MultioutputKernel, object)
@check_shapes(
    "inducing_variable_u: [M_u, D, 1]",
    "inducing_variable_v: [M_v, D, 1]",
    "Xnew: [batch..., N, D]",
    "return: [M_v, P, batch..., N, P]",
)
def Cvf_generic(
    inducing_variable_u: InducingPoints,
    inducing_variable_v: InducingPoints,
    kernel: MultioutputKernel,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return Cvf(
        inducing_variable_u,
        inducing_variable_v,
        kernel,
        Xnew,
        L_Kuu=L_Kuu,
    )  # [M, N]


@Cvf.register(
    SharedIndependentInducingVariables,
    SharedIndependentInducingVariables,
    SharedIndependent,
    object,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [M_v, batch..., N]",
)
def Cvf_shared_shared(
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: tf.Tensor,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return Cvf(
        inducing_variable_u.inducing_variable,
        inducing_variable_v.inducing_variable,
        kernel.kernel,
        Xnew,
        L_Kuu=tf.unstack(L_Kuu, axis=0),
    )  # [M_v, N]


@Cvf.register(
    SeparateIndependentInducingVariables,
    SeparateIndependentInducingVariables,
    SharedIndependent,
    object,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_separate_shared(
    inducing_variable_u: SeparateIndependentInducingVariables,
    inducing_variable_v: SeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return tf.stack(
        [
            Cvf(
                ind_var_u,
                ind_var_v,
                kernel.kernel,
                Xnew,
                L_Kuu=l_kuu,
            )
            for ind_var_u, ind_var_v, l_kuu in zip(
                inducing_variable_u.inducing_variable_list,
                inducing_variable_v.inducing_variable_list,
                tf.unstack(L_Kuu, axis=0),
            )
        ],
        axis=0,
    )


@Cvf.register(
    SharedIndependentInducingVariables,
    SharedIndependentInducingVariables,
    SeparateIndependent,
    object,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_shared_separate(
    inducing_variable_u: SharedIndependentInducingVariables,
    inducing_variable_v: SharedIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    return tf.stack(
        [
            Cvf(
                inducing_variable_u.inducing_variable,
                inducing_variable_v.inducing_variable,
                k,
                Xnew,
                L_Kuu=l_kuu,
            )
            for k, l_kuu in zip(kernel.kernels, tf.unstack(L_Kuu, axis=0))
        ],
        axis=0,
    )


@Cvf.register(
    SeparateIndependentInducingVariables,
    SeparateIndependentInducingVariables,
    SeparateIndependent,
    object,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "Xnew: [batch..., N, D]",
    "return: [L, M_v, batch..., N]",
)
def Cvf_separate_separate(
    inducing_variable_u: SeparateIndependentInducingVariables,
    inducing_variable_v: SeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    Xnew: TensorType,
    *,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    n_iv_u = len(inducing_variable_u.inducing_variable_list)
    n_iv_v = len(inducing_variable_v.inducing_variable_list)
    n_k = len(kernel.kernels)
    assert (
        n_iv_u == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv_u} and {n_k}."

    assert (
        n_iv_v == n_k
    ), f"Must have same number of inducing variables and kernels. Found {n_iv_v} and {n_k}."

    return tf.stack(
        [
            Cvf(
                ind_var_u,
                ind_var_v,
                k,
                Xnew,
                L_Kuu=l_kuu,
            )
            for k, ind_var_u, ind_var_v, l_kuu in zip(
                kernel.kernels,
                inducing_variable_u.inducing_variable_list,
                inducing_variable_v.inducing_variable_list,
                tf.unstack(L_Kuu, axis=0),
            )
        ],
        axis=0,
    )
