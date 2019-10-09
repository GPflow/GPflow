import tensorflow as tf
from ..inducing_variables import InducingPoints, Multiscale, InducingPatches
from ..kernels import Kernel, SquaredExponential, Convolutional
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


@Kuf.register(InducingPatches, Convolutional, object)
def Kuf(feat, kern, Xnew):
    Xp = kern.get_patches(Xnew)  # N x num_patches x patch_len
    bigKzx = kern.basekern.K(feat.Z, Xp)  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kern.weights if hasattr(kern, 'weights') else bigKzx, [2])
    return Kzx / kern.num_patches
