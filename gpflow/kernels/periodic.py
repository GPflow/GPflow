import numpy as np
import tensorflow as tf

from .base import Kernel
from ..base import Parameter, positive
from .stationaries import Stationary


class Periodic(Kernel):
    """
    The periodic family of kernels. Can be used to wrap any Stationary kernel
    to transform it into a periodic version. The canonical form (based on the
    SquaredExponential kernel) can be found in Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    The derivation can be achieved by mapping the original inputs through the
    transformation u = (cos(x), sin(x)).

    For the SquaredExponential base kernel, the result can be expressed as:
        k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²}

    where:
    r is the Euclidean distance between the input points
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this
    is absorbed into lengthscale hyperparameter).
    """

    def __init__(self, base, period=1.0):
        if not isinstance(base, Stationary):
            raise TypeError("Periodic requires a Stationary kernel as the `base`")
        super().__init__(base.active_dims)
        self.base = base
        self.period = Parameter(period, transform=positive())

    def K_diag(self, X, presliced=False):
        return self.base.K_diag(X)

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        if X2 is None:
            X2 = X

        self.base.validate_input_shape(X)

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now [N, 1, D]
        f2 = tf.expand_dims(X2, 0)  # now [1, M, D]

        r = np.pi * (f - f2) / self.period
        scaled_sine = tf.sin(r) / self.base.lengthscale
        if hasattr(self.base, "K_r"):
            sine_r = tf.reduce_sum(tf.abs(scaled_sine), -1)
            K = self.base.K_r(sine_r)
        else:
            sine_r2 = tf.reduce_sum(tf.square(scaled_sine), -1)
            K = self.base.K_r2(sine_r2)
        return K