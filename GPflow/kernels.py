# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import print_function, absolute_import
from functools import reduce

import tensorflow as tf
import numpy as np
from .param import Param, Parameterized, AutoFlow
from . import transforms
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class Kern(Parameterized):
    """
    The basic kernel class. Handles input_dim and active dims, and provides a
    generic '_slice' function to implement them.
    """

    def __init__(self, input_dim, active_dims=None):
        """
        input dim is an integer
        active dims is a (slice | iterable of integers | None)
        """
        Parameterized.__init__(self)
        self.scoped_keys.extend(['K', 'Kdiag'])
        self.input_dim = input_dim
        # TEMP HACK Code was using slice which cannot be handled in kernel_expectations
        #         if active_dims is None:
        #             self.active_dims = slice(input_dim)
        #         else:
        #             self._active_dims_array = np.array(active_dims, dtype=np.int32)
        #             self.active_dims = tf.constant(self._active_dims_array, tf.int32)
        if active_dims is None:
            active_dims = range(input_dim)
        self._active_dims_array = np.array(active_dims, dtype=np.int32)
        self.active_dims = tf.constant(self._active_dims_array, tf.int32)

    def _slice(self, X, X2):
        if isinstance(self.active_dims, slice):
            X = X[:, self.active_dims]
            if X2 is not None:
                X2 = X2[:, self.active_dims]
            return X, X2
        else:
            X = tf.transpose(tf.gather(tf.transpose(X), self.active_dims))
            if X2 is not None:
                X2 = tf.transpose(tf.gather(tf.transpose(X2), self.active_dims))
            return X, X2

    def __add__(self, other):
        return Add([self, other])

    def __mul__(self, other):
        return Prod([self, other])

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)

    @AutoFlow((float_type, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X)

    @AutoFlow((float_type, [None, None]))
    def compute_Kdiag(self, X):
        return self.Kdiag(X)

    def __getstate__(self):
        d = Parameterized.__getstate__(self)
        if hasattr(self, '_active_dims_array'):
            d.pop('active_dims')
        return d

    def __setstate__(self, d):
        Parameterized.__setstate__(self, d)
        if hasattr(self, '_active_dims_array'):
            self.active_dims = tf.constant(self._active_dims_array, tf.int32)


class Static(Kern):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance.
    """
    def __init__(self, input_dim, variance=1.0, active_dims=None):
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(variance, transforms.positive)

    def Kdiag(self, X):
        return tf.fill(tf.pack([tf.shape(X)[0]]), tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            d = tf.fill(tf.pack([tf.shape(X)[0]]), tf.squeeze(self.variance))
            return tf.diag(d)
        else:
            shape = tf.pack([tf.shape(X)[0], tf.shape(X2)[0]])
            return tf.zeros(shape, float_type)


class Constant(Static):
    """
    The Constant (aka Bias) kernel
    """
    def K(self, X, X2=None):
        if X2 is None:
            shape = tf.pack([tf.shape(X)[0], tf.shape(X)[0]])
        else:
            shape = tf.pack([tf.shape(X)[0], tf.shape(X2)[0]])
        return tf.fill(shape, tf.squeeze(self.variance))


class Bias(Constant):
    """
    Another name for the Constant kernel, included for convenience.
    """
    pass


class Stationary(Kern):
    """
    Base class for kernels that are stationary, that is, they only depend on

        r = || x - x' ||

    This class handles 'ARD' behaviour, which stands for 'Automatic Relevance
    Determination'. This means that the kernel has one lengthscale per
    dimension, otherwise the kernel is isotropic (has a single lengthscale).
    """
    def __init__(self, input_dim, variance=1.0, lengthscales=None,
                 active_dims=None, ARD=False):
        """
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter
        - lengthscales is the initial value for the lengthscales parameter
          defaults to 1.0 (ARD=False) or np.ones(input_dim) (ARD=True).
        - active_dims is a list of length input_dim which controls which
          columns of X are used.
        - ARD specifies whether the kernel has one lengthscale per dimension
          (ARD=True) or a single lengthscale (ARD=False).
        """
        Kern.__init__(self, input_dim, active_dims)
        self.scoped_keys.extend(['square_dist', 'euclid_dist'])
        self.variance = Param(variance, transforms.positive)
        if ARD:
            if lengthscales is None:
                lengthscales = np.ones(input_dim, np_float_type)
            else:
                # accepts float or array:
                lengthscales = lengthscales * np.ones(input_dim, np_float_type)
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
            return -2*tf.matmul(X, tf.transpose(X)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2*tf.matmul(X, tf.transpose(X2)) +\
                tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def euclid_dist(self, X, X2):
        r2 = self.square_dist(X, X2)
        return tf.sqrt(r2 + 1e-12)

    def Kdiag(self, X):
        return tf.fill(tf.pack([tf.shape(X)[0]]), tf.squeeze(self.variance))


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
        - input_dim is the dimension of the input to the kernel
        - variance is the (initial) value for the variance parameter(s)
          if ARD=True, there is one variance per input
        - active_dims is a list of length input_dim which controls
          which columns of X are used.
        """
        Kern.__init__(self, input_dim, active_dims)
        self.ARD = ARD
        if ARD:
            # accept float or array:
            variance = np.ones(self.input_dim)*variance
            self.variance = Param(variance, transforms.positive)
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


class Polynomial(Linear):
    """
    The Polynomial kernel. Samples are polynomials of degree `d`.
    """

    def __init__(self, input_dim, degree=3.0, variance=1.0, offset=1.0, active_dims=None, ARD=False):
        """
        :param input_dim: the dimension of the input to the kernel
        :param variance: the (initial) value for the variance parameter(s)
                         if ARD=True, there is one variance per input
        :param degree: the degree of the polynomial
        :param active_dims: a list of length input_dim which controls
          which columns of X are used.
        :param ARD: use variance as described
        """
        Linear.__init__(self, input_dim, variance, active_dims, ARD)
        self.degree = degree
        self.offset = Param(offset, transform=transforms.positive)

    def K(self, X, X2=None):
        return (Linear.K(self, X, X2) + self.offset) ** self.degree

    def Kdiag(self, X):
        return (Linear.Kdiag(self, X) + self.offset) ** self.degree


class Exponential(Stationary):
    """
    The Exponential kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.exp(-0.5 * r)


class Matern12(Stationary):
    """
    The Matern 1/2 kernel
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
        return self.variance * (1. + np.sqrt(3.) * r) *\
            tf.exp(-np.sqrt(3.) * r)


class Matern52(Stationary):
    """
    The Matern 5/2 kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance*(1.0 + np.sqrt(5.) * r + 5./3. * tf.square(r))\
            * tf.exp(-np.sqrt(5.) * r)


class Cosine(Stationary):
    """
    The Cosine kernel
    """
    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        r = self.euclid_dist(X, X2)
        return self.variance * tf.cos(r)


class PeriodicKernel(Kern):
    """
    The periodic kernel. Defined in  Equation (47) of

    D.J.C.MacKay. Introduction to Gaussian processes. In C.M.Bishop, editor,
    Neural Networks and Machine Learning, pages 133--165. Springer, 1998.

    Derived using the mapping u=(cos(x), sin(x)) on the inputs.
    """
    def __init__(self, input_dim, period=1.0, variance=1.0,
                 lengthscales=1.0, active_dims=None):
        # No ARD support for lengthscale or period yet
        Kern.__init__(self, input_dim, active_dims)
        self.variance = Param(variance, transforms.positive)
        self.lengthscales = Param(lengthscales, transforms.positive)
        self.ARD = False
        self.period = Param(period, transforms.positive)

    def Kdiag(self, X):
        return tf.fill(tf.pack([tf.shape(X)[0]]), tf.squeeze(self.variance))

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        if X2 is None:
            X2 = X

        # Introduce dummy dimension so we can use broadcasting
        f = tf.expand_dims(X, 1)  # now N x 1 x D
        f2 = tf.expand_dims(X2, 0)  # now 1 x M x D

        r = np.pi * (f-f2) / self.period
        r = tf.reduce_sum(tf.square(tf.sin(r)/self.lengthscales), 2)

        return self.variance * tf.exp(-0.5 * r)


class Coregion(Kern):
    def __init__(self, input_dim, output_dim, rank, active_dims=None):
        """
        A Coregionalization kernel. The inputs to this kernel are _integers_
        (we cast them from floats as needed) which usually specify the
        *outputs* of a Coregionalization model.

        The parameters of this kernel, W, kappa, specify a positive-definite
        matrix B.

          B = W W^T + diag(kappa) .

        The kernel function is then an indexing of this matrix, so

          K(x, y) = B[x, y] .

        We refer to the size of B as "num_outputs x num_outputs", since this is
        the number of outputs in a coreginoalization model. We refer to the
        number of columns on W as 'rank': it is the number of degrees of
        correlation between the outputs.

        NB. There is a symmetry between the elements of W, which creates a
        local minimum at W=0. To avoid this, it's recommended to initialize the
        optimization (or MCMC chain) using a random W.
        """
        assert input_dim == 1, "Coregion kernel in 1D only"
        Kern.__init__(self, input_dim, active_dims)

        self.output_dim = output_dim
        self.rank = rank
        self.W = Param(np.zeros((self.output_dim, self.rank)))
        self.kappa = Param(np.ones(self.output_dim), transforms.positive)

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        X = tf.cast(X[:, 0], tf.int32)
        if X2 is None:
            X2 = X
        else:
            X2 = tf.cast(X2[:, 0], tf.int32)
        B = tf.matmul(self.W, tf.transpose(self.W)) + tf.diag(self.kappa)
        return tf.gather(tf.transpose(tf.gather(B, X2)), X)

    def Kdiag(self, X):
        X, _ = self._slice(X, None)
        X = tf.cast(X[:, 0], tf.int32)
        Bdiag = tf.reduce_sum(tf.square(self.W), 1) + self.kappa
        return tf.gather(Bdiag, X)


def make_kernel_names(kern_list):
    """
    Take a list of kernels and return a list of strings, giving each kernel a
    unique name.

    Each name is made from the lower-case version of the kernel's class name.

    Duplicate kernels are given training numbers.
    """
    names = []
    counting_dict = {}
    for k in kern_list:
        raw_name = k.__class__.__name__.lower()

        # check for duplicates: start numbering if needed
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
    Combine  a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
    """
    def __init__(self, kern_list):
        for k in kern_list:
            assert isinstance(k, Kern), "can only add Kern instances"
        Kern.__init__(self, input_dim=np.max([k.input_dim for k in kern_list]))

        # add kernels to a list, flattening out instances of this class therein
        self.kern_list = []
        for k in kern_list:
            if isinstance(k, self.__class__):
                self.kern_list.extend(k.kern_list)
            else:
                self.kern_list.append(k)

        # generate a set of suitable names and add the kernels as attributes
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
