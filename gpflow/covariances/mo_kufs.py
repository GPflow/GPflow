from typing import Union

import tensorflow as tf

from ..features import (InducingPoints, MixedKernelSeparateMof,
                        MixedKernelSharedMof, SeparateIndependentMof,
                        SharedIndependentMof)
from ..kernels import (Mok, SeparateIndependentMok, SeparateMixedMok,
                       SharedIndependentMok)
from ..util import create_logger
from .dispatch import Kuf

logger = create_logger()


def debug_kuf(feature, kernel):
    msg = "Dispatch to Kuf(feature: {}, kernel: {})"
    logger.debug(
        msg.format(feature.__class__.__name__, kernel.__class__.__name__))


@Kuf.register(InducingPoints, Mok, object)
def _Kuf(feature: InducingPoints, kernel: Mok, Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    return kernel(feature.Z, Xnew, full=True,
                  full_output_cov=True)  # [M, P, N, P]


@Kuf.register(SharedIndependentMof, SharedIndependentMok, object)
def _Kuf(feature: SharedIndependentMof, kernel: SharedIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    return Kuf(feature.feature, kernel.kernel, Xnew)  # [M, N]


@Kuf.register(SeparateIndependentMof, SharedIndependentMok, object)
def _Kuf(feature: SeparateIndependentMof, kernel: SharedIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    return tf.stack([Kuf(f, kernel.kernel, Xnew) for f in feature.features],
                    axis=0)  # [L, M, N]


@Kuf.register(SharedIndependentMof, SeparateIndependentMok, object)
def _Kuf(feature: SharedIndependentMof, kernel: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    return tf.stack([Kuf(feature.feature, k, Xnew) for k in kernel.kernels],
                    axis=0)  # [L, M, N]


@Kuf.register(SeparateIndependentMof, SeparateIndependentMok, object)
def _Kuf(feature: SeparateIndependentMof, kernel: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    Kufs = [Kuf(f, k, Xnew) for f, k in zip(feature.features, kernel.kernels)]
    return tf.stack(Kufs, axis=0)  # [L, M, N]


@Kuf.register((SeparateIndependentMof, SharedIndependentMof), SeparateMixedMok,
              object)
def _Kuf(feature: Union[SeparateIndependentMof, SharedIndependentMof],
         kernel: SeparateMixedMok, Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    kuf_impl = Kuf.dispatch(type(feature), SeparateIndependentMok, object)
    K = tf.transpose(kuf_impl(feature, kernel, Xnew), [1, 0, 2])  # [M, L, N]
    return K[:, :, :, None] * tf.transpose(kernel.W)[None, :,
                                                     None, :]  # [M, L, N, P]


@Kuf.register(MixedKernelSharedMof, SeparateMixedMok, object)
def _Kuf(feature: MixedKernelSharedMof, kernel: SeparateIndependentMok,
         Xnew: tf.Tensor):
    debug_kuf(feature, kernel)
    return tf.stack([Kuf(feature.feature, k, Xnew) for k in kernel.kernels],
                    axis=0)  # [L, M, N]


@Kuf.register(MixedKernelSeparateMof, SeparateMixedMok, object)
def _Kuf(feature, kernel, Xnew):
    debug_kuf(feature, kernel)
    return tf.stack(
        [Kuf(f, k, Xnew) for f, k in zip(feature.features, kernel.kernels)],
        axis=0)  # [
    # L, M, N]
