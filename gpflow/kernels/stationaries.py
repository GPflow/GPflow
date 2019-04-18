import numpy as np
import tensorflow as tf

from gpflow.util import broadcasting_elementwise
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

    def __init__(self, variance=1.0, lengthscale=1.0, active_dims=None, ard=None):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscale is the initial value for the lengthscale parameter
          defaults to 1.0 (ard=False) or np.ones(input_dim) (ard=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - if ard is not None, it specifies whether the kernel has one
          lengthscale per dimension (ard=True) or a single lengthscale
          (ard=False). Otherwise, inferred from shape of lengthscale.
        """
        super().__init__(active_dims)
        self.ard = ard
        # lengthscale, self.ard = self._validate_ard_shape("lengthscale", lengthscale, ard)
        self.variance = Parameter(variance, transform=positive())
        self.lengthscale = Parameter(lengthscale, transform=positive())

    def scaled_square_dist(self, X, X2):
        """
        Returns ((X - X2ᵀ)/lengthscale)².
        Due to the implementation and floating-point imprecision, the
        result may actually be very slightly negative for entries very
        close to each other.
        """
        X = X / self.lengthscale

        if X2 is None:
            Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
            dist = -2 * tf.matmul(X, X, transpose_b=True)
            dist += Xs + tf.linalg.transpose(Xs)
            return dist

        Xs = tf.reduce_sum(tf.square(X), axis=-1)
        X2 = X2 / self.lengthscale
        X2s = tf.reduce_sum(tf.square(X2), axis=-1)
        dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
        dist += broadcasting_elementwise(tf.add, Xs, X2s)
        return dist

    def scaled_euclid_dist(self, X, X2):
        """
        Returns |(X - X2ᵀ)/lengthscale| (L2-norm).
        """
        r2 = self.scaled_square_dist(X, X2)
        # Clipping around the (single) float precision which is ~1e-45.
        return tf.sqrt(tf.maximum(r2, 1e-40))

    def K_diag(self, X, presliced=False):
        return tf.fill((X.shape[:-1]), tf.squeeze(self.variance))


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel,

        k(r) = σ² exp{-½ r²}

    where:
    r   is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ²  is the variance parameter

    Functions drawn from a GP with this kernel are infinitely differentiable!
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
    ℓ  : lengthscale
    α  : alpha, determines relative weighting of small-scale and large-scale fluctuations

    For α → ∞, the RQ kernel becomes equivalent to the squared exponential.
    """

    def __init__(self, variance=1.0, lengthscale=1.0, alpha=1.0, active_dims=None, ard=None):
        super().__init__(variance=variance, lengthscale=lengthscale, active_dims=active_dims,
                         ard=ard)
        self.alpha = Parameter(alpha, transform=positive())

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        distance_2 = self.scaled_square_dist(X, X2)
        return self.variance * (1 + distance_2 / (2 * self.alpha)) ** (-self.alpha)


class Exponential(Stationary):
    """
    The Exponential kernel. It is equivalent to a Matern12 kernel with doubled lengthscales.
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel. Functions drawn from a GP with this kernel are not
    differentiable anywhere. The kernel equation is

    k(r) = σ² exp{-r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ.
    σ² is the variance parameter
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.exp(-r)


class Matern32(Stationary):
    """
    The Matern 3/2 kernel. Functions drawn from a GP with this kernel are once
    differentiable. The kernel equation is

    k(r) =  σ² (1 + √3r) exp{-√3 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * (1. + np.sqrt(3.) * r) * tf.exp(-np.sqrt(3.) * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel. Functions drawn from a GP with this kernel are twice
    differentiable. The kernel equation is

    k(r) =  σ² (1 + √5r + 5/3r²) exp{-√5 r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * (1.0 + np.sqrt(5.) * r + 5. / 3. * tf.square(r)) * \
               tf.exp(-np.sqrt(5.) * r)


class Cosine(Stationary):
    """
    The Cosine kernel. Functions drawn from a GP with this kernel are sinusoids
    (with a random phase). The kernel equation is

        k(r) =  σ² cos{r}

    where:
    r  is the Euclidean distance between the input points, scaled by the lengthscale parameter ℓ,
    σ² is the variance parameter.
    """

    def K(self, X, X2=None, presliced=False):
        if not presliced:
            X, X2 = self.slice(X, X2)
        r = self.scaled_euclid_dist(X, X2)
        return self.variance * tf.cos(r)
