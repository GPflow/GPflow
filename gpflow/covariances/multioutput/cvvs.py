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

from typing import Optional, Union

import tensorflow as tf
from check_shapes import check_shapes

from gpflow.inducing_variables import (
    FallbackSeparateIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    InducingPoints,
)
from gpflow.kernels import (
    IndependentLatent,
    MultioutputKernel,
    SeparateIndependent,
    SharedIndependent,
)

from .covariances.dispatch import Cvv


@Cvv.register(InducingPoints, InducingPoints, MultioutputKernel)
@check_shapes(
    "inducing_variable_u: [M_u, D, 1]",
    "inducing_variable_v: [M_v, D, 1]",
    "return: [M_v, P, M_v, P]",
)
def Kuu_generic(
    inducing_variable_u: InducingPoints,
    inducing_variable_v: InducingPoints,
    kernel: MultioutputKernel,
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    _Cvv = Cvv(
        inducing_variable_u,
        inducing_variable_v,
        kernel,
        L_Kuu=L_Kuu,
    )  # [M, M]
    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=_Cvv.dtype) * jitter
    return _Cvv + jittermat


@Cvv.register(
    FallbackSharedIndependentInducingVariables,
    FallbackSharedIndependentInducingVariables,
    SharedIndependent,
)
def Cvv_shared_shared(
    inducing_variable_u: FallbackSharedIndependentInducingVariables,
    inducing_variable_v: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:
    _Cvv = Cvv(
        inducing_variable_u.inducing_variable,
        inducing_variable_v.inducing_variable,
        kernel.kernel,
        L_Kuu=L_Kuu,
    )  # [M, M]
    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=_Cvv.dtype) * jitter

    return _Cvv + jittermat


@Cvv.register(FallbackSharedIndependentInducingVariables, SeparateIndependent)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "return: [L, M_v, M_v]",
)
def Cvv_fallback_shared(
    inducing_variable_u: FallbackSharedIndependentInducingVariables,
    inducing_variable_v: FallbackSharedIndependentInducingVariables,
    kernel: Union[SeparateIndependent, IndependentLatent],
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    _Cvv = tf.stack(
        [
            Cvv(
                inducing_variable_u.inducing_variable,
                inducing_variable_v.inducing_variable,
                kernel.kernel,
                L_Kuu=l_kuu,
            )
            for k, l_kuu in zip(kernel.kernels, tf.unstack(L_Kuu, axis=0))
        ],
        axis=0,
    )

    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=Cvv.dtype)[None, :, :] * jitter

    return _Cvv + jittermat


@Cvv.register(
    FallbackSeparateIndependentInducingVariables,
    FallbackSeparateIndependentInducingVariables,
    SharedIndependent,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "return: [L, M_v, M_v]",
)
def Kuu_fallback_separate_shared(
    inducing_variable_u: FallbackSeparateIndependentInducingVariables,
    inducing_variable_v: FallbackSeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    _Cvv = tf.stack(
        [
            Cvv(ind_var_u, ind_var_v, kernel.kernel, L_Kuu=l_kuu)
            for ind_var_u, ind_var_v, l_kuu in zip(
                inducing_variable_u.inducing_variable_list,
                inducing_variable_v.inducing_variable_list,
                tf.unstack(L_Kuu, axis=0),
            )
        ],
        axis=0,
    )

    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=Cvv.dtype)[None, :, :] * jitter

    return _Cvv + jittermat


@Cvv.register(
    FallbackSeparateIndependentInducingVariables,
    FallbackSeparateIndependentInducingVariables,
    SeparateIndependent,
)
@check_shapes(
    "inducing_variable_u: [M_u, D, P]",
    "inducing_variable_v: [M_v, D, P]",
    "return: [L, M_v, M_v]",
)
def Kuu_fallback_separate(
    inducing_variable_u: FallbackSeparateIndependentInducingVariables,
    inducing_variable_v: FallbackSeparateIndependentInducingVariables,
    kernel: SeparateIndependent,
    *,
    jitter: float = 0.0,
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

    lista = []
    for ind_var_u, ind_var_v, l_kuu, k in zip(
        inducing_variable_u.inducing_variable_list,
        inducing_variable_v.inducing_variable_list,
        tf.unstack(L_Kuu, axis=0),
        kernel.kernels,
    ):

        lista.append(Cvv(ind_var_u, ind_var_v, k, L_Kuu=l_kuu))

    """
    lista =  [
>>>>>>> 53dae0f597e5aa12eea4bc13839a3c94807acd36
            Cvv(ind_var_u, ind_var_v, k, L_Kuu=l_kuu)
            for ind_var_u, ind_var_v, l_kuu, k in zip(
                inducing_variable_u.inducing_variable_list,
                inducing_variable_v.inducing_variable_list,
                L_Kuu,
                kernel.kernels,
            )
        ]
    """

    _Cvv = tf.stack(
        lista,
        axis=0,
    )

    jittermat = tf.eye(inducing_variable_v.num_inducing, dtype=_Cvv.dtype)[None, :, :] * jitter

    return _Cvv + jittermat
