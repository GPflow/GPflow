from typing import Union

import tensorflow as tf

from ..features import (InducingPoints, SharedIndependentInducingVariablesBase,
                        SeparateIndependentInducingVariablesBase)
from ..kernels import (Mok, SeparateIndependentMok, SeparateMixedMok,
                       SharedIndependentMok)
from .dispatch import Kuu


@Kuu.register(InducingPoints, Mok)
def _Kuu(feature: InducingPoints, kernel: Mok, *, jitter=0.0):
    Kmm = kernel(feature.Z, full=True, full_output_cov=True)  # [M, P, M, P]
    M = Kmm.shape[0] * Kmm.shape[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), Kmm.shape)
    return Kmm + jittermat


@Kuu.register(SharedIndependentInducingVariablesBase, SharedIndependentMok)
def _Kuu(feature: SharedIndependentInducingVariablesBase,
         kernel: SharedIndependentMok,
         *,
         jitter=0.0):
    Kmm = Kuu(feature.feature, kernel.kernel)  # [M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(SharedIndependentInducingVariablesBase, (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feature: SharedIndependentInducingVariablesBase,
         kernel: Union[SeparateIndependentMok, SeparateMixedMok],
         *,
         jitter=0.0):
    Kmm = tf.stack([Kuu(feature.feature, k) for k in kernel.kernels],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentInducingVariablesBase, SharedIndependentMok)
def _Kuu(feature: SeparateIndependentInducingVariablesBase,
         kernel: SharedIndependentMok,
         *,
         jitter=0.0):
    Kmm = tf.stack([Kuu(f, kernel.kernel) for f in feature.features],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentInducingVariablesBase,
              (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feature: SeparateIndependentInducingVariablesBase,
         kernel: Union[SeparateIndependentMok, SeparateMixedMok],
         *,
         jitter=0.0):
    Kmms = [Kuu(f, k) for f, k in zip(feature.features, kernel.kernels)]
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
