import numpy as np
import tensorflow as tf
epsilon = 1e-9


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
        return np.log( 1. + np.exp(x) )
    def tf_forward(self, x):
        #clip x to prevent overflow
        x = tf.clip_by_value(x, -np.inf, 300)
        one = 0. * x + 1. # ensures shape
        return tf.log( one + tf.exp(x) ) + epsilon
    def tf_log_jacobian(self, x):
        return -tf.reduce_sum(tf.log(1. + tf.exp(-x)))
    def backward(self, y):
        return np.log(np.exp(y) - np.ones(1))
    def __str__(self):
        return '+ve'

positive = Log1pe()


