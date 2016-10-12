# Copyright 2016 James Hensman, alexggmatthews
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

from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from . import tf_wraps as tfw
from ._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


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
        Return the log Jacobian of the tf_forward mapping.

        Note that we *could* do this using a tf manipulation of
        self.tf_forward, but tensorflow may have difficulty: it doesn't have a
        Jacaobian at time of writing.  We do this in the tests to make sure the
        implementation is correct.
        """
        raise NotImplementedError

    def free_state_size(self, variable_shape):
        return np.prod(variable_shape)

    def __str__(self):
        """
        A short string describing the nature of the constraint
        """
        raise NotImplementedError

    def __getstate__(self):
        return self.__dict__.copy()

    def __setstate__(self, d):
        self.__dict__ = d


class Identity(Transform):
    def tf_forward(self, x):
        return tf.identity(x)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def tf_log_jacobian(self, x):
        return tf.zeros((1,), float_type)

    def __str__(self):
        return '(none)'


class Exp(Transform):
    def __init__(self, lower=1e-6):
        self._lower = lower

    def tf_forward(self, x):
        return tf.exp(x) + self._lower

    def forward(self, x):
        return np.exp(x) + self._lower

    def backward(self, y):
        return np.log(y - self._lower)

    def tf_log_jacobian(self, x):
        return tf.reduce_sum(x)

    def __str__(self):
        return '+ve'


class Log1pe(Transform):
    """
    A transform of the form

       y = \log ( 1 + \exp(x))

    x is a free variable, y is always positive.

    This function is known as 'softplus' in tensorflow.
    """

    def __init__(self, lower=1e-6):
        """
        lower is a float that defines the minimum value that this transform can
        take, default 1e-6. This helps stability during optimization, because
        aggressive optimizers can take overly-long steps which lead to zero in
        the transformed variable, causing an error.
        """
        self._lower = lower

    def forward(self, x):
        return np.log(1. + np.exp(x)) + self._lower

    def tf_forward(self, x):
        return tf.nn.softplus(x) + self._lower

    def tf_log_jacobian(self, x):
        return -tf.reduce_sum(tf.nn.softplus(-x))

    def backward(self, y):
        return np.log(np.exp(y - self._lower) - np.ones(1, np_float_type))

    def __str__(self):
        return '+ve'


class Logistic(Transform):
    def __init__(self, a=0., b=1.):
        Transform.__init__(self)
        assert b > a
        self.a, self.b = a, b
        self._a = tf.constant(a, float_type)
        self._b = tf.constant(b, float_type)

    def tf_forward(self, x):
        ex = tf.exp(-x)
        return self._a + (self._b - self._a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def tf_log_jacobian(self, x):
        return tf.reduce_sum(x - 2. * tf.log(tf.exp(x) + 1.) + tf.log(self._b - self._a))

    def __str__(self):
        return '[' + str(self.a) + ', ' + str(self.b) + ']'

    def __getstate__(self):
        d = Transform.__getstate__(self)
        d.pop('_a')
        d.pop('_b')
        return d

    def __setstate__(self, d):
        Transform.__setstate__(self, d)
        self._a = tf.constant(self.a, float_type)
        self._b = tf.constant(self.b, float_type)


class LowerTriangular(Transform):
    """
    A transform of the form

       tri_mat = vec_to_tri(x)

    x is a free variable, y is always a list of lower triangular matrices sized
    (N x N x D).
    """

    def __init__(self, num_matrices=1, squeeze=False):
        """
        Create an instance of LowerTriangular transform.
        Args:
            num_matrices: Number of matrices to be stored.
            squeeze: If num_matrices == 1, drop the redundant axis.
        """
        self.num_matrices = num_matrices  # We need to store this for reconstruction.
        self.squeeze = squeeze

    def _validate_vector_length(self, length):
        """
        Check whether the vector length is consistent with being a triangular
         matrix and with `self.num_matrices`.
        Args:
            length: Length of the free state vector.

        Returns: Length of the vector with the lower triangular elements.

        """
        L = length / self.num_matrices
        if int(((L * 8) + 1) ** 0.5) ** 2.0 != (L * 8 + 1):
            raise ValueError("The free state must be a triangle number.")
        return L

    def forward(self, x):
        """
        Transforms from the free state to the variable.
        Args:
            x: Free state vector. Must have length of `self.num_matrices` *
                triangular_number.

        Returns:
            Reconstructed variable.
        """
        L = self._validate_vector_length(len(x))
        matsize = int((L * 8 + 1) ** 0.5 * 0.5 - 0.5)
        xr = np.reshape(x, (self.num_matrices, -1))
        var = np.zeros((matsize, matsize, self.num_matrices), np_float_type)
        for i in range(self.num_matrices):
            indices = np.tril_indices(matsize, 0)
            var[indices + (np.zeros(len(indices[0])).astype(int) + i,)] = xr[i, :]
        return var.squeeze() if self.squeeze else var

    def backward(self, y):
        """
        Transforms from the variable to the free state.
        Args:
            y: Variable representation.

        Returns:
            Free state.
        """
        N = int((y.size / self.num_matrices) ** 0.5)
        y = np.reshape(y, (N, N, self.num_matrices))
        return y[np.tril_indices(len(y), 0)].T.flatten()

    def tf_forward(self, x):
        fwd = tf.transpose(tfw.vec_to_tri(tf.reshape(x, (self.num_matrices, -1))), [1, 2, 0])
        return tf.squeeze(fwd) if self.squeeze else fwd

    def tf_log_jacobian(self, x):
        return tf.zeros((1,), float_type) - np.inf

    def free_state_size(self, variable_shape):
        matrix_batch = len(variable_shape) > 2
        if ((not matrix_batch and self.num_matrices != 1) or
           (matrix_batch and variable_shape[2] != self.num_matrices)):
            raise ValueError("Number of matrices must be consistent with what was passed to the constructor.")
        if variable_shape[0] != variable_shape[1]:
            raise ValueError("Matrices passed must be square.")
        N = variable_shape[0]
        return int(0.5 * N * (N + 1)) * (variable_shape[2] if matrix_batch else 1)

    def __str__(self):
        return "LoTri->vec"


class BoundedBelow(Log1pe):
    """
    A transform of the form

       y = \log ( 1 + \exp(x)) + alpha

    x is a free variable, y > alpha
    """

    def __init__(self, alpha=1.0):
        """
        alpha is a float that defines the minimum value that this transform can
        take, default 1. note that alpha might also be a np.array that matches
        the shape of the parameter.
        """
        Log1pe.__init__(self, alpha)

    def __str__(self):
        return '>{}'.format(self._lower)


class BoundedAbove(Transform):
    """
    A transform of the form

       y = \\alpha - \\log ( 1 + \\exp(x))

    x is a free variable, y < alpha
    """

    def __init__(self, alpha=1.0):
        """
        alpha is a float that defines the minimum value that this transform can
        take, default 1. note that alpha might also be a np.array that matches
        the shape of the parameter.
        """
        self.alpha = alpha

    def forward(self, x):
        return self.alpha - np.log(1. + np.exp(x))

    def tf_forward(self, x):
        return self.alpha - tf.nn.softplus(x)

    def tf_log_jacobian(self, x):
        return -tf.reduce_sum(tf.nn.softplus(-x))

    def backward(self, y):
        return np.log(np.exp(self.alpha - y) - np.ones(1, np_float_type))

    def __str__(self):
        return '<{}'.format(self._lower)


positive = Log1pe()
