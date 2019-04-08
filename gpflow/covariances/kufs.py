import tensorflow as tf
from ..features import InducingPoints, Multiscale
from ..kernels import Kernel, RBF
from .dispatch import Kuf


@Kuf.register(InducingPoints, Kernel, object)
def _Kuf(feat: InducingPoints, kern: Kernel, Xnew: tf.Tensor):
    return kern(feat.Z, Xnew)


@Kuf.register(Multiscale, RBF, object)
def _Kuf(feat: Multiscale, kern: RBF, Xnew):
    Xnew, _ = kern.slice(Xnew, None)
    Zmu, Zlen = kern.slice(feat.Z, feat.scales)
    idlengthscale = kern.lengthscale + Zlen
    d = feat._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kern.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kern.variance * tf.exp(-d / 2) * lengthscale)
