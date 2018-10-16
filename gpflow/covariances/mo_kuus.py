from typing import Union

import tensorflow as tf

from ..features import (InducingPoints, MixedKernelSharedMof,
                        SeparateIndependentMof, SharedIndependentMof)
from ..kernels import (Mok, SeparateIndependentMok, SeparateMixedMok,
                       SharedIndependentMok)
from ..util import create_logger
from .dispatch import kuu_

logger = create_logger()


def debug_kuu(feat, kern, jitter):
    msg = "Dispatch to Kuu(feat: {}, kern: {}) with jitter={}"
    logger.debug(msg.format(
        feat.__class__.__name__,
        kern.__class__.__name__,
        jitter))


@Kuu.register(InducingPoints, Mok)
def _Kuu(feat: InducingPoints,
        kern: Mok, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = kern(feat.Z, full_output_cov=True)  # M x P x M x P
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=Kmm.dtype), tf.shape(Kmm))
    return Kmm + jittermat


@Kuu.register(SharedIndependentMof, SharedIndependentMok)
def _Kuu(feat: SharedIndependentMof,
        kern: SharedIndependentMok, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = Kuu(feat.feat, kern.kern)  # M x M
    jittermat = tf.eye(len(feat), dtype=Kmm.dtype) * jitter
    return Kmm + jittermat


@Kuu.register(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feat: SharedIndependentMof,
        kern: Union[SeparateIndependentMok, SeparateMixedMok], *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kernels], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentMof, SharedIndependentMok)
def _Kuu(feat: SeparateIndependentMof,
        kern: SharedIndependentMok, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(f, kern.kern) for f in feat.feat_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(SeparateIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def _Kuu(feat: SeparateIndependentMof,
        kern: Union[SeparateIndependentMok, SeparateMixedMok], *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kernels)], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat


@Kuu.register(MixedKernelSharedMof, SeparateMixedMok)
def _Kuu(feat: MixedKernelSharedMof,
        kern: SeparateMixedMok, *, jitter=0.0):
    debug_kuu(feat, kern, jitter)
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kernels], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=Kmm.dtype)[None, :, :] * jitter
    return Kmm + jittermat