import tensorflow as tf
from ..base import Parameter, positive
from .base import Kernel


class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
    """

    def __init__(self, variance=1.0, active_dims=None):
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())

    def K_diag(self, X, presliced=False):
        return tf.fill((X.shape[0], ), tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            d = tf.fill((X.shape[0], ), tf.squeeze(self.variance))
            return tf.linalg.diag(d)
        else:
            shape = [X.shape[0], X2.shape[0]]
            return tf.zeros(shape, dtype=X.dtype)


class Constant(Static):
    """
    The Constant (aka Bias) kernel. Functions drawn from a GP with this kernel
    are constant, i.e. f(x) = c, with c ~ N(0, σ^2). The kernel equation is

        k(x, y) = σ²

    where:
    σ²  is the variance parameter.
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            shape = tf.stack([X.shape[0], X.shape[0]])
        else:
            shape = tf.stack([X.shape[0], X2.shape[0]])
        return tf.fill(shape, tf.squeeze(self.variance))
