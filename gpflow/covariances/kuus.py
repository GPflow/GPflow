import tensorflow as tf
from ..features import InducingPoints, Multiscale
from ..kernels import Kernel, RBF
from .dispatch import Kuu


@Kuu.register(InducingPoints, Kernel)
def _Kuu(feat: InducingPoints, kern: Kernel, *, jitter=0.0):
    Kzz = kern(feat.Z())
    Kzz += jitter * tf.eye(len(feat), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, RBF)
def _Kuu(feat: Multiscale, kern: RBF, *, jitter=0.0):
    Zmu, Zlen = kern.slice(feat.Z(), feat.scales())
    idlengthscales2 = tf.square(kern.lengthscales() + Zlen)
    sc = tf.sqrt(
        tf.expand_dims(idlengthscales2, 0) + tf.expand_dims(idlengthscales2, 1) - tf.square(
            kern.lengthscales()))
    d = feat._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kern.variance() * tf.exp(-d / 2) * tf.reduce_prod(kern.lengthscales() / sc, 2)
    Kzz += jitter * tf.eye(len(feat), dtype=Kzz.dtype)
    return Kzz
