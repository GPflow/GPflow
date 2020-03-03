import tensorflow as tf

from ..base import Parameter
from ..utilities import positive
from .base import Kernel


class Linear(Kernel):
    """
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx.
    The kernel equation is

        k(x, y) = σ²xy

    where σ² is the variance parameter.
    """

    def __init__(self, variance=1.0, active_dims=None):
        """
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())
        self._validate_ard_active_dims(self.variance)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        return self.variance.shape.ndims > 0

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X

        return tf.linalg.matmul(X * self.variance, X2, transpose_b=True)

    def K_diag(self, X):
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
    def __init__(self, degree=3.0, variance=1.0, offset=1.0, active_dims=None):
        """
        :param degree: the degree of the polynomial
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param offset: the offset of the polynomial
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(variance, active_dims)
        self.degree = degree
        self.offset = Parameter(offset, transform=positive())

    def K(self, X, X2=None):
        return (super().K(X, X2) + self.offset)**self.degree

    def K_diag(self, X):
        return (super().K_diag(X) + self.offset)**self.degree
