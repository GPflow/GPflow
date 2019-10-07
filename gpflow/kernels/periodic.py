import numpy as np
import tensorflow as tf

from .base import Kernel
from ..base import Parameter, positive


class Periodic(Kernel):
    """
    The periodic kernel. Defined in  Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    Derived using an RBF kernel once mapped the original inputs through
    the mapping u=(cos(x), sin(x)).

    The resulting periodic kernel can be expressed as:
        k(r) =  σ² exp{ -0.5 sin²(π r / γ) / ℓ²}

    where:
    r  is the Euclidean distance between the input points
    ℓ is the lengthscale parameter,
    σ² is the variance parameter,
    γ is the period parameter.

    (note that usually we have a factor of 4 instead of 0.5 in front but this is absorbed into lengthscale
    hyperparameter).
    """

    def __init__(self,
                 period=1.0,
                 variance=1.0,
                 lengthscale=1.0,
                 active_dims=None):
        # No ard support for lengthscale or period yet
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())
        self.ard = False
        self.period = Parameter(period, transform=positive())

    def K_diag(self, X, presliced=False):
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now [N, 1, D]
        f2 = tf.expand_dims(X2, 0)  # now [1, M, D]

        r = np.pi * (f - f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r) / self.lengthscale), 2)

        return self.variance * tf.exp(-0.5 * r)