import tensorflow as tf

from ..inducing_variables import InducingPoints, Multiscale
from ..kernels import Kernel, SquaredExponential
from .dispatch import Kuu


@Kuu.register(InducingPoints, Kernel)
def _Kuu(inducing_variable: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, SquaredExponential)
def _Kuu(inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter=0.0):
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale2 = tf.square(kernel.lengthscale + Zlen)
    sc = tf.sqrt(idlengthscale2[None, ...] + idlengthscale2[:, None, ...] -
                 kernel.lengthscale**2)
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(
        kernel.lengthscale / sc, 2)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz
