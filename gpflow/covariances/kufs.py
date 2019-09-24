import tensorflow as tf
from ..base import TensorLike
from ..inducing_variables import InducingVariables, InducingPoints, Multiscale
from ..kernels import Kernel, SquaredExponential
from .dispatch import Kuf


@Kuf.register(InducingVariables, Kernel, TensorLike)
def Kuf_fallback(inducing_variable, kernel, X):
    from warnings import warn
    warn('Kuf(inducing_variable, kernel, X) is deprecated, please use '
         'Kuf(kernel, inducing_variable, X) instead.',
         DeprecationWarning)
    return Kuf(kernel, inducing_variable, X)


@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)


@Kuf.register(Multiscale, SquaredExponential, TensorLike)
def Kuf_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale = kernel.lengthscale + Zlen
    d = inducing_variable._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kernel.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-0.5 * d) * lengthscale)
