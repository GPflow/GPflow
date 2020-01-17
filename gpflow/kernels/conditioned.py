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

    def __init__(self, base: Kernel, x_c: tf.Tensor, y_c: tf.Tensor):
        """
        :param base: the base kernel to make conditioned; must inherit from Kernel
        :param x_c: conditioning input
        :param y_c: conditioning output
        """

        if not isinstance(base, Kernel):
            raise TypeError("Conditioned requires a Kernel object as the `base`")

        super().__init__()
        self.base = base
        self.x_c = x_c
        self.y_c = y_c
        self.num_c = x_c.shape[0]

    @property
    def L_c(self):
        """
        The Cholesky factor of the Covariance at the conditioning inputs K(Xc, Xc)
        """
        K_c = self.base.K(self.x_c) + \
            tf.eye(self.num_c, dtype=default_float()) * default_jitter()
        return tf.linalg.cholesky(K_c)

    def K_diag(self, X: tf.Tensor, presliced: bool = False) -> tf.Tensor:
        K_Xc = self.base.K(self.x_c, X)
        return self.base.K_diag(X) - \
               tf.reduce_sum(tf.square(tf.linalg.triangular_solve(self.L_c, K_Xc)), axis=-2)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None, presliced: bool = False) -> tf.Tensor:
        K_Xc = self.base.K(self.x_c, X)
        U = tf.linalg.triangular_solve(self.L_c, K_Xc)
        if X2 is None:
            return self.base.K(X) - tf.matmul(U, U, transpose_a=True)
        else:
            K_X2c = self.base.K(self.x_c, X2)
            U2 = tf.linalg.triangular_solve(self.L_c, K_X2c)
            return self.base.K(X) - tf.matmul(U, U2, transpose_a=True)

    def conditional_mean(self, X: tf.Tensor):
        K_Xc = self.base.K(X, self.x_c)
        return K_Xc @ tf.linalg.cholesky_solve(self.L_c, self.y_c)
