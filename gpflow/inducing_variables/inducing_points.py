import tensorflow as tf
from ..base import TensorLike
from ..covariances import Kuf, Kuu
from ..kernels import Kernel
from .base import InducingPointsBase


class InducingPoints(InducingPointsBase):
    """
    Real-space inducing points
    """


@Kuu.register(InducingPoints, Kernel)
def Kuu_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, *, jitter=0.0):
    Kzz = kernel(inducing_variable.Z)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuf.register(InducingPoints, Kernel, TensorLike)
def Kuf_kernel_inducingpoints(inducing_variable: InducingPoints, kernel: Kernel, Xnew):
    return kernel(inducing_variable.Z, Xnew)
