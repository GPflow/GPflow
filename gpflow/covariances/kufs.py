import tensorflow as tf
from ..inducing_variables import InducingPoints, Multiscale
from ..kernels import Kernel, SquaredExponential
from .dispatch import Kuf


@Kuf.register(InducingPoints, Kernel, object)
def _Kuf(inducing_variable: InducingPoints, kernel: Kernel, Xnew: tf.Tensor):
    return kernel(inducing_variable.Z, Xnew)


@Kuf.register(Multiscale, SquaredExponential, object)
def _Kuf(inducing_variable: Multiscale, kernel: SquaredExponential, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale = kernel.lengthscale + Zlen
    d = inducing_variable._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kernel.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-0.5 * d) * lengthscale)
