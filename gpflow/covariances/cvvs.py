#
# Copyright (c) 2022 The GPflux Contributors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from typing import Optional

import tensorflow as tf

from gpflow.config import default_jitter
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel

from gpflux.covariances.dispatch import Cvv


@Cvv.register(InducingPoints, InducingPoints, Kernel)
def Cvv_kernel_inducingpoints(
    inducing_variable_u: InducingPoints,
    inducing_variable_v: InducingPoints,
    kernel: Kernel,
    *,
    jitter: float = 0.0,
    L_Kuu: Optional[tf.Tensor] = None,
) -> tf.Tensor:

    Kvv = kernel(inducing_variable_v.Z)

    if L_Kuu is None:
        Kuu = kernel(inducing_variable_u.Z)
        jittermat = tf.eye(inducing_variable_u.num_inducing, dtype=Kuu.dtype) * default_jitter()
        Kuu += jittermat
        L_Kuu = tf.linalg.cholesky(Kuu)

    Kuv = kernel(inducing_variable_u.Z, inducing_variable_v.Z)

    L_Kuu_inv_Kuv = tf.linalg.triangular_solve(L_Kuu, Kuv)
    Cvv = Kvv - tf.linalg.matmul(L_Kuu_inv_Kuv, L_Kuu_inv_Kuv, transpose_a=True)

    Cvv += jitter * tf.eye(inducing_variable_v.num_inducing, dtype=Cvv.dtype)

    return Cvv
