import tensorflow as tf

from ..inducing_variables import InducingPoints, Multiscale, InducingPatches
from ..kernels import Kernel, SquaredExponential, Convolutional
from .dispatch import Kuu
from ..config import default_float


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(Multiscale, SquaredExponential)
def Kuu_squaredexponential_multiscale(
    inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter=0.0
):
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscales2 = tf.square(kernel.lengthscales + Zlen)
    sc = tf.sqrt(
        idlengthscales2[None, ...] + idlengthscales2[:, None, ...] - kernel.lengthscales ** 2
    )
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(kernel.lengthscales / sc, 2)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuu.register(InducingPatches, Convolutional)
def Kuu_conv_patch(inducing_variable: InducingPatches, kernel: Convolutional, jitter=0.0):
    return kernel.base_kernel.K(inducing_variable.Z) + jitter * tf.eye(
        len(inducing_variable), dtype=default_float()
    )
