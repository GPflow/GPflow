from typing import Union

import tensorflow as tf

from ..inducing_variables import (InducingPoints, FallbackSharedIndependentInducingVariables,
                                  FallbackSeparateIndependentInducingVariables)
from ..kernels import (MultioutputKernel, SeparateIndependent, LinearCoregionalisation,
                       SharedIndependent)
from .dispatch import Kuu


@Kuu.register(InducingPoints, MultioutputKernel)
def _Kuu(feature: InducingPoints, kernel: MultioutputKernel, *, jitter=0.0):
    Kmm = kernel(feature.Z, full=True, full_output_cov=True)  # [M, P, M, P]
    M = Kmm.shape[0] * Kmm.shape[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), Kmm.shape)
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, SharedIndependent)
def _Kuu(feature: FallbackSharedIndependentInducingVariables,
         kernel: SharedIndependent,
         *,
         jitter=0.0):
    Kmm = Kuu(feature.inducing_variable, kernel.kernel)  # [M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSharedIndependentInducingVariables, (SeparateIndependent, LinearCoregionalisation))
def _Kuu(feature: FallbackSharedIndependentInducingVariables,
         kernel: Union[SeparateIndependent, LinearCoregionalisation],
         *,
         jitter=0.0):
    Kmm = tf.stack([Kuu(feature.inducing_variable, k) for k in kernel.kernels],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables, SharedIndependent)
def _Kuu(feature: FallbackSeparateIndependentInducingVariables,
         kernel: SharedIndependent,
         *,
         jitter=0.0):
    Kmm = tf.stack([Kuu(f, kernel.kernel) for f in feature.inducing_variable_list],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(FallbackSeparateIndependentInducingVariables,
              (SeparateIndependent, LinearCoregionalisation))
def _Kuu(feature: FallbackSeparateIndependentInducingVariables,
         kernel: Union[SeparateIndependent, LinearCoregionalisation],
         *,
         jitter=0.0):
    Kmms = [Kuu(f, k) for f, k in zip(feature.inducing_variable_list, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat

# @Kuu.register(MixedKernelSharedMof, SeparateMixedMok)
# def _Kuu(feature: MixedKernelSharedMof,
#          kernel: SeparateMixedMok,
#          *,
#          jitter=0.0):
#     Kmm = tf.stack([Kuu(feature.feature, k) for k in kernel.kernels],
#                    axis=0)  # [L, M, M]
#     jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
#     return Kmm + jittermat
