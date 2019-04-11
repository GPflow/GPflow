import tensorflow as tf
from ..features import InducingPoints, Multiscale
from ..kernels import Kernel, RBF
from .dispatch import Kuf


@Kuf.register(InducingPoints, Kernel, object)
def _Kuf(feature: InducingPoints, kernel: Kernel, Xnew: tf.Tensor):
    return kernel(feature.Z, Xnew)


@Kuf.register(Multiscale, RBF, object)
def _Kuf(feature: Multiscale, kernel: RBF, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(feature.Z, feature.scales)
    idlengthscale = kernel.lengthscale + Zlen
    d = feature._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kernel.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-d / 2) * lengthscale)
