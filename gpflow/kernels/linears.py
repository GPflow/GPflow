import tensorflow as tf
from ..base import Parameter, positive
from .base import Kernel


class Linear(Kernel):
    """
    The linear kernel
    """

    def __init__(self, variance=1.0, active_dims=None, ard=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ard=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(active_dims)

        # variance, self.ard = self._validate_ard_shape("variance", variance, ard)
        self.ard = ard
        self.variance = Parameter(variance, transform=positive())

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)

        if X2 is None:
            return tf.matmul(X * self.variance, X, transpose_b=True)

        return tf.matmul(X * self.variance, X2, transpose_b=True)

    def K_diag(self, X, presliced=False):
        if not presliced:
            X, _ = self.slice(X, None)
        return tf.reduce_sum(tf.square(X) * self.variance, 1)


class Polynomial(Linear):
    """
    The Polynomial kernel. Samples are polynomials of degree `d`.
    """

    def __init__(self,
                 degree=3.0,
                 variance=1.0,
                 offset=1.0,
                 active_dims=None,
                 ard=None):
        """
        :param input_dim: the dimension of the input to the kernel
        :param variance: the (initial) value for the variance parameter(s)
                         if ard=True, there is one variance per input
        :param degree: the degree of the polynomial
        :param active_dims: a list of length input_dim which controls
          which columns of X are used.
        :param ard: use variance as described
        """
        super().__init__(variance, active_dims, ard)
        self.degree = degree
        self.offset = Parameter(offset, transform=positive())

    def K(self, X, X2=None, presliced=False):
        return (Linear(self, X, X2, presliced=presliced) + self.offset) ** self.degree

    def K_diag(self, X, presliced=False):
        return (Linear(self, X, presliced=presliced) + self.offset) ** self.degree

