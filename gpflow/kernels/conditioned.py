from typing import Optional, Tuple
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

    def __init__(self, base: Kernel, data_cond: Tuple[tf.Tensor, tf.Tensor]):
        """
        :param base: the base kernel to make conditioned; must inherit from Kernel
        :param xc: conditioning input
        :param yc: conditioning output
        """

        if not isinstance(base, Kernel):
            raise TypeError("Conditioned requires a Kernel object as the `base`")

        super().__init__()
        self.base = base
        self.data_cond = data_cond
        self.X_cond, self.Y_cond = data_cond
        self.num_cond = self.X_cond.shape[0]

    @property
    def chol_K_cond(self):
        """
        The Cholesky factor of the Covariance at the conditioning inputs K(Xc, Xc)
        """
        K_cond = self.base.K(self.X_cond) + \
                 tf.eye(self.num_cond, dtype=default_float()) * default_jitter()
        return tf.linalg.cholesky(K_cond)

    def K_diag(self, X: tf.Tensor, presliced: bool = False) -> tf.Tensor:
        K_condx = self.base.K(self.X_cond, X)
        U = tf.linalg.triangular_solve(self.chol_K_cond, K_condx)
        return self.base.K_diag(X) - tf.reduce_sum(tf.square(U), axis=-2)

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None, presliced: bool = False) -> tf.Tensor:
        K_condx = self.base.K(self.X_cond, X)
        U_condx = tf.linalg.triangular_solve(self.chol_K_cond, K_condx)
        if X2 is None:
            return self.base.K(X) - tf.matmul(U_condx, U_condx, transpose_a=True)
        else:
            K_condx2 = self.base.K(self.X_cond, X2)
            U_condx2 = tf.linalg.triangular_solve(self.chol_K_cond, K_condx2)
            return self.base.K(X, X2) - tf.matmul(U_condx, U_condx2, transpose_a=True)

    def conditional_mean(self, X: tf.Tensor):
        K_xcond = self.base.K(X, self.X_cond)
        return K_xcond @ tf.linalg.cholesky_solve(self.chol_K_cond, self.Y_cond)
