import tensorflow as tf

from ..features import InducingPoints, Multiscale
from ..kernels import Kernel, RBF
from .dispatch import Kuu


@Kuu.register(InducingPoints, Kernel)
def _Kuu(feature: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(feature.Z)
    Kzz += jitter * tf.eye(len(feature), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, RBF)
def _Kuu(feature: Multiscale, kernel: RBF, *, jitter=0.0):
    Zmu, Zlen = kernel.slice(feature.Z, feature.scales)
    idlengthscale2 = tf.square(kernel.lengthscale + Zlen)
    sc = tf.sqrt(idlengthscale2[None, ...] + idlengthscale2[:, None, ...] -
                 kernel.lengthscale**2)
    d = feature._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(
        kernel.lengthscale / sc, 2)
    Kzz += jitter * tf.eye(len(feature), dtype=Kzz.dtype)
    return Kzz
