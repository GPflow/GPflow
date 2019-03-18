import numpy as np
import tensorflow as tf
from ..base import Parameter, positive
from .base import Kernel


class Stationary(Kernel):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ard' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """

    def __init__(self, variance=1.0, lengthscales=1.0, active_dims=None, ard=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ard=False) or np.ones(input_dim) (ard=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - if ard is not None, it specifies whether the kernel has one
          lengthscale per dimension (ard=True) or a single lengthscale
          (ard=False). Otherwise, inferred from shape of lengthscales.
        """
        super().__init__(active_dims)
        self.ard = ard
        # lengthscales, self.ard = self._validate_ard_shape("lengthscales", lengthscales, ard)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscales = Parameter(lengthscales, transform=positive())


    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscales)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), axis=1)

        if X2 is None:
            dist = -2 * tf.linalg.matmul(X, X, transpose_b=True)
            dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
            return dist

        X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), axis=1)
        dist = -2 * tf.linalg.matmul(X, X2, transpose_b=True)
        dist += tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
        return dist


    def scaled_euclid_dist(self, X, X2):
        """
        Returns |(X - X2ᵀ)/lengthscales| (L2-norm).
        """
        r2 = self.scaled_square_dist(X, X2)
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(tf.maximum(r2, 1e-40))


    def K_diag(self, X, presliced=False):
        return tf.fill(tf.stack([X.shape[0]]), tf.squeeze(self.variance))


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        return self.variance * tf.exp(-self.scaled_square_dist(X, X2) / 2)


class RationalQuadratic(Stationary):
    """
    Rational Quadratic kernel,

    k(r) = σ² (1 + r² / 2αℓ²)^(-α)

    σ² : variance
    ℓ  : lengthscales
    α  : alpha, determines relative weighting of small-scale and large-scale fluctuations

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """

    def __init__(self, variance=1.0, lengthscales=1.0, alpha=1.0, active_dims=None, ard=None):
        super().__init__(variance=variance, lengthscales=lengthscales, active_dims=active_dims, ard=ard)
        self.alpha = Parameter(alpha, transform=positive())

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        return self.variance * (1 + self.scaled_square_dist(X, X2) / (2 * self.alpha)) ** (-self.alpha)


class Exponential(Stationary):
    """
    The Exponential kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.exp(-r)


class Matern32(Stationary):
    """
    The Matern 3/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * (1. + np.sqrt(3.) * r) * tf.exp(-np.sqrt(3.) * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * (1.0 + np.sqrt(5.) * r + 5. / 3. * tf.square(r)) * tf.exp(-np.sqrt(5.) * r)


class Cosine(Stationary):
    """
    The Cosine kernel
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.cos(r)

