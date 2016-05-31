import numpy as np
import tensorflow as tf


class Transform(object):
    def forward(self, x):
        """
        Map from the free-space to the variable space, using numpy
        """
        raise NotImplementedError

    def backward(self, y):
        """
        Map from the variable-space to the free space, using numpy
        """
        raise NotImplementedError

    def tf_forward(self, x):
        """
        Map from the free-space to the variable space, using tensorflow
        """
        raise NotImplementedError

    def tf_log_jacobian(self, x):
        """
        Return the log jacobian of the tf_forward mapping.

        Note that we *could* do this using a tf manipluation of
        self.tf_forward, but tensorflow may have difficulty: it doesn't have a
        jacaobian at time of writing.  We do this in the tests to make sure the
        implementation is correct.
        """
        raise NotImplementedError

    def __str__(self):
        """
        A short string desscribing the nature of the constraint
        """
        raise NotImplementedError


class Identity(Transform):
    def tf_forward(self, x):
        return tf.identity(x)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def tf_log_jacobian(self, x):
        return tf.zeros((1,), tf.float64)

    def __str__(self):
        return '(none)'


class Exp(Transform):
    def tf_forward(self, x):
        return tf.exp(x)

    def forward(self, x):
        return np.exp(x)

    def backward(self, y):
        return np.log(y)

    def tf_log_jacobian(self, x):
        return tf.reduce_sum(x)

    def __str__(self):
        return '+ve'


class Log1pe(Transform):
    """
    A transform of the form

       y = \log ( 1 + \exp(x))

    x is a free variable, y is always positive.
    """
    def forward(self, x):
        return np.log(1. + np.exp(x))

    def tf_forward(self, x):
        one = 0. * x + 1.  # ensures shape
        return tf.log(one + tf.exp(x))

    def tf_log_jacobian(self, x):
        return -tf.reduce_sum(tf.log(1. + tf.exp(-x)))

    def backward(self, y):
        return np.log(np.exp(y) - np.ones(1))

    def __str__(self):
        return '+ve'


class Logistic(Transform):
    def __init__(self, a=0., b=1.):
        Transform.__init__(self)
        assert b > a
        self.a, self.b = a, b
        self._a = tf.constant(a, tf.float64)
        self._b = tf.constant(b, tf.float64)

    def tf_forward(self, x):
        ex = tf.exp(-x)
        return self._a + (self._b-self._a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b-self.a) / (1. + ex)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def tf_log_jacobian(self, x):
        return tf.reduce_sum(x - 2. * tf.log(tf.exp(x) + 1.) + tf.log(self._b - self._a))

    def __str__(self):
        return '[' + str(self.a) + ', ' + str(self.b) + ']'


positive = Log1pe()
