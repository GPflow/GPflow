from typing import Union

import tensorflow as tf

from ...inducing_variables import (
    InducingPoints,
    FallbackSharedIndependentInducingVariables,
    FallbackSeparateIndependentInducingVariables,
    SharedIndependentInducingVariables,
)
from ...kernels import (
    MultioutputKernel,
    SeparateIndependent,
    LinearCoregionalization,
    SharedIndependent,
    IndependentLatent,
)
from ..dispatch import Kuu


@Kuu.register(InducingPoints, MultioutputKernel)
def _Kuu(inducing_variable: InducingPoints, kernel: MultioutputKernel, *, jitter=0.0):
    Kmm = kernel(inducing_variable.Z, full_cov=True, full_output_cov=True)  # [M, P, M, P]
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), tf.shape(Kmm))
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def _Kuu(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter=0.0,
):
    Kmm = Kuu(inducing_variable.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, (SeparateIndependent, IndependentLatent))
def _Kuu(
    inducing_variable: FallbackSharedIndependentInducingVariables,
    kernel: Union[SeparateIndependent, IndependentLatent],
    *,
    jitter=0.0,
):
    Kmm = tf.stack(
        [Kuu(inducing_variable.inducing_variable, k) for k in kernel.kernels], axis=0
    )  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables, SharedIndependent)
def _Kuu(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: SharedIndependent,
    *,
    jitter=0.0,
):
    Kmm = tf.stack(
        [Kuu(f, kernel.kernel) for f in inducing_variable.inducing_variable_list], axis=0
    )  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(
    FallbackSeparateIndependentInducingVariables, (SeparateIndependent, LinearCoregionalization)
)
def _Kuu(
    inducing_variable: FallbackSeparateIndependentInducingVariables,
    kernel: Union[SeparateIndependent, LinearCoregionalization],
    *,
    jitter=0.0,
):
    Kmms = [Kuu(f, k) for f, k in zip(inducing_variable.inducing_variable_list, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(len(inducing_variable), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat
