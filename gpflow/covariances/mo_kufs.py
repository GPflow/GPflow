from typing import Union

import tensorflow as tf

from ..features import (InducingPoints, MixedKernelSharedMof,
                        SeparateIndependentMof, SharedIndependentMof)
from ..kernels import (Mok, SeparateIndependentMok, SeparateMixedMok,
                       SharedIndependentMok)
from ..util import create_logger
from .dispatch import Kuf

logger = create_logger()


def debug_kuf(feat, kern):
    msg = "Dispatch to Kuf(feat: {}, kern: {})"
    logger.debug(msg.format(
        feat.__class__.__name__,
        kern.__class__.__name__))

@Kuf.register(InducingPoints, Mok, object)
def _Kuf(feat: InducingPoints,
         kern: Mok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    return kern(feat.Z(), Xnew, full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentMof, SharedIndependentMok, object)
def _Kuf(feat: SharedIndependentMof,
         kern: SharedIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    return Kuf(feat.feat, kern.kern, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentMof, SharedIndependentMok, object)
def _Kuf(feat: SeparateIndependentMof,
         kern: SharedIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(f, kern.kern, Xnew) for f in feat.feat_list], axis=0)  # [L, M, N]


@Kuf.register(SharedIndependentMof, SeparateIndependentMok, object)
def _Kuf(feat: SharedIndependentMof,
         kern: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kernels], axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentMof, SeparateIndependentMok, object)
def _Kuf(feat: SeparateIndependentMof,
         kern: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    Kufs = [Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kernels)]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register((SeparateIndependentMof, SharedIndependentMof), SeparateMixedMok, object)
def _Kuf(feat: Union[SeparateIndependentMof, SharedIndependentMof],
         kern: SeparateMixedMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    kuf_impl = Kuf.dispatch(type(feat), SeparateIndependentMok, object)
    K = tf.transpose(kuf_impl(feat, kern, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kern.W())[None, :, None, :]  # [M, L, N, P]


@Kuf.register(MixedKernelSharedMof, SeparateMixedMok, object)
def _Kuf(feat: MixedKernelSharedMof,
         kern: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feat, kern)
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kernels], axis=0)  # [L, M, N]
