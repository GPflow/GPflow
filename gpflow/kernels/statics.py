import tensorflow as tf

from ..base import Parameter
from ..utilities import positive
from .base import Kernel


class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
    """

    def __init__(self, variance=1.0, active_dims=None):
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())

    def K_diag(self, X):
        return tf.fill((tf.shape(X)[0],), tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """

    def K(self, X, X2=None):
        if X2 is None:
            d = tf.fill((tf.shape(X)[0],), tf.squeeze(self.variance))
            return tf.linalg.diag(d)
        else:
            shape = [tf.shape(X)[0], tf.shape(X2)[0]]
            return tf.zeros(shape, dtype=X.dtype)


class Constant(Static):
    """
    The Constant (aka Bias) kernel. Functions drawn from a GP with this kernel
    are constant, i.e. f(x) = c, with c ~ N(0, σ^2). The kernel equation is

        k(x, y) = σ²

    where:
    σ²  is the variance parameter.
    """

    def K(self, X, X2=None):
        if X2 is None:
            shape = [tf.shape(X)[0], tf.shape(X)[0]]
        else:
            shape = [tf.shape(X)[0], tf.shape(X2)[0]]
        return tf.fill(shape, tf.squeeze(self.variance))
