import tensorflow as tf
from ..base import Parameter, positive
from .base import Kernel


class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """

    def __init__(self, variance=1.0, active_dims=None):
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())

    def K_diag(self, X, presliced=False):
        return tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            d = tf.fill(tf.stack([tf.shape(X)[0]]), tf.squeeze(self.variance))
            return tf.matrix_diag(d)
        else:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X2)[0]])
            return tf.zeros(shape, dtype=X.dtype)


class Constant(Static):
    """
    The Constant (aka Bias) kernel
    """

    def K(self, X, X2=None, presliced=False):
        if X2 is None:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X)[0]])
        else:
            shape = tf.stack([tf.shape(X)[0], tf.shape(X2)[0]])
        return tf.fill(shape, tf.squeeze(self.variance))