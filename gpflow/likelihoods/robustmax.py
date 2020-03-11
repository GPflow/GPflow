import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from ..base import Module, Parameter
from ..config import default_int
from ..utilities import to_default_float, to_default_int


class RobustMax(Module):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-epsilon)  i == argmax(f)
          epsilon/(k-1)  otherwise

    where k is the number of classes.
    """

    def __init__(self, num_classes, epsilon=1e-3, **kwargs):
        """
        `epsilon` represents the fraction of 'errors' in the labels of the
        dataset. This may be a hard parameter to optimize, so by default
        it is set un-trainable, at a small value.
        """
        super().__init__(**kwargs)
        transform = tfp.bijectors.Sigmoid()
        prior = tfp.distributions.Beta(to_default_float(0.2), to_default_float(5.0))
        self.epsilon = Parameter(epsilon, transform=transform, prior=prior, trainable=False)
        self.num_classes = num_classes
        self._squash = 1e-6

    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(
            i, self.num_classes, tf.squeeze(1.0 - self.epsilon), tf.squeeze(self.eps_k1)
        )

    @property
    def eps_k1(self):
        return self.epsilon / (self.num_classes - 1.0)

    def safe_sqrt(self, val):
        return tf.sqrt(tf.clip_by_value(val, 1e-10, np.inf))

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = to_default_int(Y)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1.0, 0.0), dtype=mu.dtype
        )
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            self.safe_sqrt(2.0 * var_selected), (-1, 1)
        )

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            self.safe_sqrt(var), 2
        )
        cdfs = 0.5 * (1.0 + tf.math.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2 * self._squash) + self._squash

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(
            tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0.0, 1.0), dtype=mu.dtype
        )
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.reduce_prod(cdfs, axis=[1]) @ tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1))
