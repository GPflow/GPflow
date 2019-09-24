import tensorflow as tf

from ..inducing_variables import InducingVariables, InducingPoints, Multiscale
from ..kernels import Kernel, SquaredExponential
from .dispatch import Kuu


@Kuu.register(InducingVariables, Kernel)
def Kuu_fallback(inducing_variable, kernel, **kw):
    from warnings import warn
    warn('Kuu(inducing_variable, kernel) is deprecated, please use '
         'Kuu(kernel, inducing_variable) instead.',
         DeprecationWarning)
    return Kuu(kernel, inducing_variable, **kw)


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, SquaredExponential)
def Kuu_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter=0.0):
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale2 = tf.square(kernel.lengthscale + Zlen)
    sc = tf.sqrt(idlengthscale2[None, ...] + idlengthscale2[:, None, ...] -
                 kernel.lengthscale**2)
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(
        kernel.lengthscale / sc, 2)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz
