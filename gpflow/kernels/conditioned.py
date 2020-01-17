from typing import Optional
import tensorflow as tf
from .base import Kernel
from ..config import default_float, default_jitter


class Conditioned(Kernel):
    """
    Conditioned kernel. Can be used to wrap any kernel
    to transform it into a its conditioned version.

    This provides a simple way to condition a Gaussian process on noiseless observations

    From a covariance function k(.,.) and a conditioning dataset (Xc, Yc),
    a new covariance function and a conditional mean function
    are created as

       kc(.,.) = k(.,.) - k(.,Xc)k(Xc, Xc)⁻¹k(Xc,.)
       mc(.)   = k(.,Xc)k(Xc, Xc)⁻¹Yc
    """

    def __init__(self, base: Kernel, xc: tf.Tensor, yc: tf.Tensor):
        """
        :param base: the base kernel to make conditioned; must inherit from Kernel
        :param xc: conditioning input
        :param yc: conditioning output
        """

        if not isinstance(base, Kernel):
            raise TypeError("Conditioned requires a Kernel object as the `base`")

        super().__init__()
        self.base = base
        self.xc = xc
        self.yc = yc
        self.num_c = xc.shape[0]

    @property
    def Lc(self):
        """
        The Cholesky factor of the Covariance at the conditioning inputs K(Xc, Xc)
        """
        Kc = self.base.K(self.xc) + \
            tf.eye(self.num_c, dtype=default_float()) * default_jitter()
        return tf.linalg.cholesky(Kc)

    def K_diag(self, X: tf.Tensor, presliced: bool = False) -> tf.Tensor:
        Kcx = self.base.K(self.xc, X)
        return self.base.K_diag(X) - \
               tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self.Lc, Kcx)), axis=-2)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None, presliced: bool = False) -> tf.Tensor:
        Kcx = self.base.K(self.xc, X)
        U = tf.linalg.triangular_solve(self.Lc, Kcx)
        if X2 is None:
            return self.base.K(X) - tf.matmul(U, U, transpose_a=True)
        else:
            Kcx2 = self.base.K(self.xc, X2)
            U2 = tf.linalg.triangular_solve(self.Lc, Kcx2)
            return self.base.K(X) - tf.matmul(U, U2, transpose_a=True)

    def conditional_mean(self, X: tf.Tensor):
        Kxc = self.base.K(X, self.xc)
        return Kxc @ tf.linalg.cholesky_solve(self.Lc, self.yc)
