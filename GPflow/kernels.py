from functools import reduce

import tensorflow as tf
from .tf_hacks import eye
import numpy as np
from .param import Param, Parameterized
from . import transforms

class Kern(Parameterized):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.

    input dim is an integer
    active dims is a (slice | iterable of integers | None)
    """
    def __init__(self, input_dim, active_dims=None):
        Parameterized.__init__(self)
        self.input_dim = input_dim
        if active_dims is None:
            self.active_dims = slice(input_dim)
        else:
            self.active_dims = active_dims

    def _slice(self, X, X2):
        if isinstance(self.active_dims, slice):
            X = X[:,self.active_dims]
            if X2 is not None:
                X2 = X2[:,self.active_dims]
            return X, X2
        else: # TODO: when tf can do fancy indexing, use that.
            X = tf.transpose(tf.pack([X[:,i] for i in self.active_dims]))
            if X2 is not None:
                X2 = tf.transpose(tf.pack([X2[:,i] for i in self.active_dims]))
            return X, X2

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])


class Static(Kern):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """
    def __init__(self, input_dim, variance=1.0, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(variance, transforms.positive)
    def Kdiag(self, X):
        zeros = X[:,0]*0
        return zeros + self.variance


class White(Static):
    """
    The White kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            return self.variance * eye(tf.shape(X)[0])
        else:
            return tf.zeros(tf.pack([tf.shape(X)[0], tf.shape(X2)[0]]), tf.float64)


class Bias(Static):
    """
    The Bias (constant) kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            return self.variance * tf.ones(tf.pack([tf.shape(X)[0], tf.shape(X)[0]]), tf.float64)
        else:
            return self.variance * tf.ones(tf.pack([tf.shape(X)[0], tf.shape(X2)[0]]), tf.float64)


class Stationary(Kern):
    """
    Base class for kernels that are statinoary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None, active_dims=None, ARD=False):
        """
        input_dim is the dimension of the input to the kernel
        variance is the (initial) value for the variance parameter
        lengthscales is the initial value for the lengthscales parameter
         --defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        active_dims is a list of length input_dim which controls thwich columns of X are used.
        ARD specified whether the kernel has one lengthscale per dimension (ARD=True) or a single lengthscale (ARD=False).
        """
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(variance, transforms.positive)
        if ARD:
            if lengthscales is None:
                lengthscales = np.ones(input_dim)
            else:
                lengthscales = lengthscales * np.ones(input_dim) # accepts float or array
            self.lengthscales = Param(lengthscales, transforms.positive)
            self.ARD = True
        else:
            if lengthscales is None:
                lengthscales = 1.0
            self.lengthscales = Param(lengthscales, transforms.positive)
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
    def __init__(self, input_dim, variance=1.0, active_dims=None, ARD=False):
        """
        input_dim is the dimension of the input to the kernel
        variance is the (initial) value for the variance parameter(s)
         -- if ARD=True, there is one variance per input
        active_dims is a list of length input_dim which controls which columns of X are used.
        """
        Kern.__init__(self, input_dim, active_dims)
        self.ARD = ARD
        if ARD:
            self.variance = Param(np.ones(self.input_dim)*variance, transforms.positive)
        else:
            self.variance = Param(variance, transforms.positive)
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


def make_kernel_names(kern_list):
    """
    Take a list of kernels and return a list of strings, giving each kernel a
    unique name.

    Each name is made from the lower-case version of the kernel's class name.

    Duplicate kernels are given training numbers.
    """
    names = []
    counting_dict ={}
    for k in kern_list:
        raw_name = k.__class__.__name__.lower()

        #check for duplicates: start numbering if needed
        if raw_name in counting_dict:
            if counting_dict[raw_name] == 1:
                names[names.index(raw_name)] = raw_name + '_1'
            counting_dict[raw_name] += 1
            name = raw_name + '_' + str(counting_dict[raw_name])
        else:
            counting_dict[raw_name] = 1
            name = raw_name
        names.append(name)
    return names



class Combination(Kern):
    """
    Combine  a list of kernels, e.g. by adding or multiplying (see inherriting classes).

    The names of the kernels to be combined are generated from their class names.
    """
    def __init__(self, kern_list):
        for k in kern_list:
            assert isinstance(k, Kern), "can only add Kern instances"
        Kern.__init__(self, input_dim=np.max([k.input_dim for k in kern_list]))

        #add kernels to a list, flattening out instances of this class therein.
        self.kern_list = []
        for k in kern_list:
            if isinstance(k, self.__class__):
                self.kern_list.extend(k.kern_list)
            else:
                self.kern_list.append(k)

        #generate a set of suitable names and add the kernels as atributes of this one.
        names = make_kernel_names(self.kern_list)
        [setattr(self, name, k) for name, k in zip(names, self.kern_list)]

class Add(Combination):
    def K(self, X, X2=None):
        return reduce(tf.add, [k.K(X, X2) for k in self.kern_list])

    def Kdiag(self, X):
        return reduce(tf.add, [k.Kdiag(X) for k in self.kern_list])


class Prod(Combination):
    def K(self, X, X2=None):
        return reduce(tf.mul, [k.K(X, X2) for k in self.kern_list])

    def Kdiag(self, X):
        return reduce(tf.mul, [k.Kdiag(X) for k in self.kern_list])
