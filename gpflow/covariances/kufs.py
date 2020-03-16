import tensorflow as tf
from ..base import TensorLike
from ..inducing_variables import InducingPoints, Multiscale, InducingPatches
from ..kernels import Kernel, SquaredExponential, Convolutional
from .dispatch import Kuf


@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)


@Kuf.register(Multiscale, SquaredExponential, TensorLike)
def Kuf_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscales = kernel.lengthscales + Zlen
    d = inducing_variable._cust_square_dist(Xnew, Zmu, idlengthscales)
    lengthscales = tf.reduce_prod(kernel.lengthscales / idlengthscales, 1)
    lengthscales = tf.reshape(lengthscales, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-0.5 * d) * lengthscales)


@Kuf.register(InducingPatches, Convolutional, object)
def Kuf_conv_patch(feat, kern, Xnew):
    Xp = kern.get_patches(Xnew)  # [N, num_patches, patch_len]
    bigKzx = kern.base_kernel.K(feat.Z, Xp)  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kern.weights if hasattr(kern, "weights") else bigKzx, [2])
    return Kzx / kern.num_patches
