import tensorflow as tf
from ..base import Parameter, TensorLike
from ..covariances import Kuf, Kuu
from ..kernels import SquaredExponential
from ..utilities import positive
from .base import InducingPointsBase


class Multiscale(InducingPointsBase):
    r"""
    Multi-scale inducing variables

    Originally proposed in

    ::

      @incollection{NIPS2009_3876,
        title = {Inter-domain Gaussian Processes for Sparse Inference using Inducing Features},
        author = {Miguel L\'{a}zaro-Gredilla and An\'{\i}bal Figueiras-Vidal},
        booktitle = {Advances in Neural Information Processing Systems 22},
        year = {2009},
      }
    """
    def __init__(self, Z, scales):
        super().__init__(Z)
        # Multi-scale inducing_variable widths (std. dev. of Gaussian)
        self.scales = Parameter(scales, transform=positive())
        if self.Z.shape != scales.shape:
            raise ValueError("Input locations `Z` and `scales` must have the same shape.")  # pragma: no cover

    @staticmethod
    def _cust_square_dist(A, B, sc):
        """
        Custom version of _square_dist that allows sc to provide per-datapoint length
        scales. sc: [N, M, D].
        """
        return tf.reduce_sum(tf.square((tf.expand_dims(A, 1) - tf.expand_dims(B, 0)) / sc), 2)
@Kuu.register(Multiscale, SquaredExponential)
def Kuu_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, *, jitter=0.0):
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale2 = tf.square(kernel.lengthscale + Zlen)
    sc = tf.sqrt(idlengthscale2[None, ...] + idlengthscale2[:, None, ...] -
                 kernel.lengthscale ** 2)
    d = inducing_variable._cust_square_dist(Zmu, Zmu, sc)
    Kzz = kernel.variance * tf.exp(-d / 2) * tf.reduce_prod(
        kernel.lengthscale / sc, 2)
    Kzz += jitter * tf.eye(len(inducing_variable), dtype=Kzz.dtype)
    return Kzz


@Kuf.register(Multiscale, SquaredExponential, TensorLike)
def Kuf_sqexp_multiscale(inducing_variable: Multiscale, kernel: SquaredExponential, Xnew):
    Xnew, _ = kernel.slice(Xnew, None)
    Zmu, Zlen = kernel.slice(inducing_variable.Z, inducing_variable.scales)
    idlengthscale = kernel.lengthscale + Zlen
    d = inducing_variable._cust_square_dist(Xnew, Zmu, idlengthscale)
    lengthscale = tf.reduce_prod(kernel.lengthscale / idlengthscale, 1)
    lengthscale = tf.reshape(lengthscale, (1, -1))
    return tf.transpose(kernel.variance * tf.exp(-0.5 * d) * lengthscale)
