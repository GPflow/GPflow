import tensorflow as tf
from ..base import Parameter, positive
from .base import Kernel


class Linear(Kernel):
    """
    The linear kernel.  Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx.
    The kernel equation is

        k(x, y) = σ²xy

    where σ²  is the variance parameter.
    """

    def __init__(self, variance=1.0, active_dims=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          defaults to 1.0 but supports ARD i.e. one variance per input.
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())

    @property
    def ard(self):
        """
        Indicates if ARD behavior is on i.e. a separate variance per dimension.
        """
        return tf.rank(self.variance) > 0

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)

        self.validate_ard_parameter(X, self.variance)

        if X2 is None:
            return tf.linalg.matmul(X * self.variance, X, transpose_b=True)

        return tf.linalg.matmul(X * self.variance, X2, transpose_b=True)

    def K_diag(self, X, presliced=False):
        if not presliced:
            X, _ = self.slice(X, None)
        return tf.reduce_sum(tf.square(X) * self.variance, 1)


class Polynomial(Linear):
    """
    The Polynomial kernel. Functions drawn from a GP with this kernel are
    polynomials of degree `d`. The kernel equation is

        k(x, y) = (σ²xy + γ)ᵈ

    where:
    σ² is the variance parameter,
    γ is the offset parameter,
    d is the degree parameter.
    """

    def __init__(self,
                 degree=3.0,
                 variance=1.0,
                 offset=1.0,
                 active_dims=None):
        """
        :param input_dim: the dimension of the input to the kernel
        :param variance: the (initial) value for the variance parameter(s)
                         defaults to 1.0 but supports ARD i.e. one variance per input.
        :param degree: the degree of the polynomial
        :param active_dims: a list of length input_dim which controls
                            which columns of X are used.
        """
        super().__init__(variance, active_dims)
        self.degree = degree
        self.offset = Parameter(offset, transform=positive())

    def K(self, X, X2=None, presliced=False):
        return (super().K(X, X2, presliced=presliced) +
                self.offset)**self.degree

    def K_diag(self, X, presliced=False):
        return (super().K_diag(X, presliced=presliced) +
                self.offset)**self.degree
