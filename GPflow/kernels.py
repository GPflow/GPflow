import tensorflow as tf
import numpy as np
from param import Param, Parameterized
import transforms

class Kern(Parameterized):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.

    input dim is an integer
    active dims is a (slice | tf integer array | numpy integer array)
    """
    def __init__(self, input_dim, active_dims=None):
        Parameterized.__init__(self)
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = slice(input_dim)
        else:
            self.active_dims = active_dims
    def _slice(self, X, X2):
        X = X[:,self.active_dims]
        if X2 is not None:
            X2 = X2[:,self.active_dims]
        return X, X2    
    def __add__(self, other):
        return Add(self, other)


class Static(Kern):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """
    def __init__(self, input_dim, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(1., transforms.positive)
    def Kdiag(self, X):
        zeros = X[:,0]*0 
        return zeros + self.variance


class White(Static):
    """
    The White kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            return self.variance * tf.eye(X.get_shape()[0])
        else:
            return self.variance * tf.zeros((X.shape[0], X2.shape[0]))


class Bias(Static):
    """
    The Bias (constant) kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            return self.variance * tf.ones((X.shape[0], X.shape[0]), tf.float64)
        else:
            return self.variance * tf.ones((X.shape[0], X2.shape[0]), tf.float64)


class Stationary(Kern):
    """
    Base class for kernels that are statinoary, that is, they only depend on 

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale). 
    """
    def __init__(self, input_dim, active_dims=None, ARD=False):
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(1., transforms.positive)
        if ARD:
            self.lengthscales = Param(np.ones(self.input_dim), transforms.positive)
            self.ARD = True
        else:
            self.lengthscales = Param(np.ones(1), transforms.positive)
            self.ARD = False

    def square_dist(self, X, X2):
        X = X/self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2*tf.matmul(X, tf.transpose(X)) + tf.reshape(Xs, (-1,1)) + tf.reshape(Xs, (1,-1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2*tf.matmul(X, tf.transpose(X2)) + tf.reshape(Xs, (-1,1)) + tf.reshape(X2s, (1,-1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        zeros = X[:,0]*0 
        return zeros + self.variance


class RBF(Stationary):
    """
    The radial basis function (RBF) or squared exponential kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        return self.variance * tf.exp(-self.square_dist(X, X2)/2)


class Linear(Kern):
    """
    The linear kernel
    """
    def __init__(self, input_dim, active_dims=None, ARD=False):
        Kern.__init__(self, input_dim, active_dims)
        self.ARD = ARD
        if ARD:
            self.variance = Param(np.ones(self.input_dim), transforms.positive)
        else:
            self.variance = Param(np.ones(1), transforms.positive)
        self.parameters = [self.variance]    

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        if X2 is None:
            return tf.matmul(X * self.variance, tf.transpose(X))
        else:
            return tf.matmul(X * self.variance, tf.transpose(X2))

    def Kdiag(self, X):
        return tf.reduce_sum(tf.square(X) * self.variance, 1)


class Exponential(Stationary):
    """
    The Exponential kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-0.5 * r)


class OU(Stationary):
    """
    The Ornstein Uhlenbeck kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-r)


class Matern32(Stationary):
    """
    The Matern 3/2 kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * (1. + np.sqrt(3.) * r) * tf.exp(-np.sqrt(3.) * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance*(1+np.sqrt(5.)*r+5./3*tf.square(r))*tf.exp(-np.sqrt(5.)*r)


class Cosine(Stationary):
    """
    The Cosine kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.cos(r)


class Add(Kern):
    """
    Add two kernels together.

    NB. We don't add multiple kernels, prefering to nest instances of this
    object. Hopefully Theano should take care of any efficiency issues.
    """
    def __init__(self, k1, k2):
        assert isinstance(k1, Kern) and isinstance(k2, Kern), "can only add Kern instances"
        Kern.__init__(self, input_dim=max(k1.input_dim, k2.input_dim))
        self.k1, self.k2 = k1, k2
    def K(self, X, X2=None):
        return self.k1.K(X, X2) + self.k2.K(X, X2)
    def Kdiag(self, X):
        return self.k1.Kdiag(X) + self.k2.Kdiag(X)


