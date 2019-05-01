from typing import Union

import tensorflow as tf

from ..features import (InducingPoints, MixedKernelSharedMof,
                        SeparateIndependentMof, SharedIndependentMof)
from ..kernels import (Mok, SeparateIndependentMok, SeparateMixedMok,
                       SharedIndependentMok)
from ..util import create_logger
from .dispatch import Kuu

logger = create_logger()


def debug_kuu(feature, kernel, jitter):
    msg = "Dispatch to Kuu(feature: {}, kernel: {}) with jitter={}"
    logger.debug(
        msg.format(feature.__class__.__name__, kernel.__class__.__name__,
                   jitter))


@Kuu.register(InducingPoints, Mok)
def _Kuu(feature: InducingPoints, kernel: Mok, *, jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmm = kernel(feature.Z, full=True, full_output_cov=True)  # [M, P, M, P]
    M = Kmm.shape[0] * Kmm.shape[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), Kmm.shape)
    return Kmm + jittermat


@Kuu.register(SharedIndependentMof, SharedIndependentMok)
def _Kuu(feature: SharedIndependentMof,
         kernel: SharedIndependentMok,
         *,
         jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmm = Kuu(feature.feature, kernel.kernel)  # [M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feature: SharedIndependentMof,
         kernel: Union[SeparateIndependentMok, SeparateMixedMok],
         *,
         jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmm = tf.stack([Kuu(feature.feature, k) for k in kernel.kernels],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentMof, SharedIndependentMok)
def _Kuu(feature: SeparateIndependentMof,
         kernel: SharedIndependentMok,
         *,
         jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmm = tf.stack([Kuu(f, kernel.kernel) for f in feature.features],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentMof,
              (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feature: SeparateIndependentMof,
         kernel: Union[SeparateIndependentMok, SeparateMixedMok],
         *,
         jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmms = [Kuu(f, k) for f, k in zip(feature.features, kernel.kernels)]
    Kmm = tf.stack(Kmms, axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(MixedKernelSharedMof, SeparateMixedMok)
def _Kuu(feature: MixedKernelSharedMof,
         kernel: SeparateMixedMok,
         *,
         jitter=0.0):
    debug_kuu(feature, kernel, jitter)
    Kmm = tf.stack([Kuu(feature.feature, k) for k in kernel.kernels],
                   axis=0)  # [L, M, M]
    jittermat = tf.eye(len(feature), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat
