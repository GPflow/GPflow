import tensorflow as tf
from ..base import TensorLike
from ..covariances import Kuf, Kuu
from ..kernels import Convolutional
from . import InducingPoints


class InducingPatches(InducingPoints):
    pass



@Kuu.register(InducingPatches, Convolutional)
def Kuu_conv_patch(feat, kern, jitter=0.0):
    return kern.basekern.K(feat.Z) + jitter * tf.eye(len(feat), dtype=feat.Z.dtype)


@Kuf.register(InducingPatches, Convolutional, TensorLike)
def Kuf_conv_patch(feat, kern, Xnew):
    Xp = kern.get_patches(Xnew)  # N x num_patches x patch_len
    bigKzx = kern.basekern.K(feat.Z, Xp)  # [M, N, P] -- thanks to broadcasting of kernels
    Kzx = tf.reduce_sum(bigKzx * kern.weights if hasattr(kern, 'weights') else bigKzx, [2])
    return Kzx / kern.num_patches
