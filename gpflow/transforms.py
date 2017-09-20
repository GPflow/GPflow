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
    """
    The identity transform: y = x
    """
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
    """
    The exponential transform:

       y = \exp(x) + \epsilon

    x is a free variable, y is always positive. The epsilon value (self.lower)
    prevents the optimizer reaching numerical zero.
    """
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
    .. math::

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
        """
        Implementation of softplus. Overflow avoided by use of the logaddexp function.
        self._lower is added before returning.
        """
        return np.logaddexp(0, x) + self._lower

    def tf_forward(self, x):
        return tf.nn.softplus(x) + self._lower

    def tf_log_jacobian(self, x):
        return -tf.reduce_sum(tf.log(1. + tf.exp(-x)))

    def backward(self, y):
        """
        Inverse of the softplus transform:
        .. math::

           x = \log( \exp(y) - 1)

        The bound for the input y is [self._lower. inf[, self._lower is
        subtracted prior to any calculations. The implementation avoids overflow
        explicitly by applying the log sum exp trick:
        .. math::

           \log ( \exp(y) - \exp(0)) &= ys + \log( \exp(y-ys) - \exp(-ys)) \\
                                     &= ys + \log( 1 - \exp(-ys)

           ys = \max(0, y)

        As y can not be negative, ys could be replaced with y itself.
        However, in case :math:`y=0` this results in np.log(0). Hence the zero is
        replaced by a machine epsilon.
        .. math::

           ys = \max( \epsilon, y)


        """
        ys = np.maximum(y - self._lower, np.finfo(np_float_type).eps)
        return ys + np.log(-np.expm1(-ys))

    def __str__(self):
        return '+ve'


class Logistic(Transform):
    """
    The logictic transform, useful for keeping variables constrained between the limits a and b:
    .. math::

       y = a + (b-a) s(x)
       s(x) = 1 / (1 + \exp(-x))
   """
    def __init__(self, a=0., b=1.):
        Transform.__init__(self)
        assert b > a
        self.a, self.b = float(a), float(b)

    def tf_forward(self, x):
        ex = tf.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def tf_log_jacobian(self, x):
        return tf.reduce_sum(x - 2. * tf.log(tf.exp(x) + 1.) + np.log(self.b - self.a))

    def __str__(self):
        return '[' + str(self.a) + ', ' + str(self.b) + ']'


class Rescale(Transform):
    """
    A transform that can linearly rescale parameters, in conjucntion with
    another transform. By default, the identity transform is wrapped so
    .. math::
       y = factor * x

    If another transform t() is passed to the constructor, then this transform becomes
    .. math::
       y = factor * t(x)

    This is useful for avoiding optimization or MCMC over large or small scales.
    """
    def __init__(self, factor=1.0, chain_transform=Identity()):
        self.factor = factor
        self.chain_transform = chain_transform

    def tf_forward(self, x):
        return self.chain_transform.tf_forward(x * self.factor)

    def forward(self, x):
        return self.chain_transform.forward(x * self.factor)

    def backward(self, y):
        return self.chain_transform.backward(y) / self.factor

    def tf_log_jacobian(self, x):
        return tf.cast(tf.reduce_prod(tf.shape(x)), float_type) * \
                self.factor * self.chain_transform.tf_log_jacobian(x * self.factor)

    def __str__(self):
        return "R" + self.chain_transform.__str__()


class DiagMatrix(Transform):
    """
    A transform to represent diagonal matrices.

    The output of this transform is a N x dim x dim array of diagonal matrices.
    The contructor argumnet dim specifies the size of the matrixes.

    Additionally, to ensure that the matrices are positive definite, the
    diagonal elements are pushed through a 'positive' transform, defaulting to
    log1pe.
    """

    def __init__(self, dim=1, positive_transform=Log1pe()):
        self.dim = dim
        self._lower = 1e-6
        self._positive_transform = positive_transform

    def forward(self, x):
        # Create diagonal matrix
        x = self._positive_transform.forward(x).reshape((-1, self.dim))
        m = np.zeros((x.shape[0], x.shape[1], x.shape[1]))
        m[(np.s_[:],) + np.diag_indices(x.shape[1])] = x
        return m

    def backward(self, y):
        # Return diagonals of matrices
        return self._positive_transform.backward(y.reshape(-1, self.dim, self.dim).diagonal(0, 1, 2).flatten())

    def tf_forward(self, x):
        return tf.matrix_diag(tf.reshape(self._positive_transform.tf_forward(x), (-1, self.dim)))

    def tf_log_jacobian(self, x):
        return tf.zeros((1,), float_type) + self._positive_transform.tf_log_jacobian(x)

    def __str__(self):
        return 'DiagMatrix'

    def free_state_size(self, variable_shape):
        return variable_shape[0] * variable_shape[1]


class LowerTriangular(Transform):
    """
    A transform of the form

       tri_mat = vec_to_tri(x)

    x is a free variable, y is always a list of lower triangular matrices sized
    (N x N x D).
    """

    def __init__(self, N, num_matrices=1, squeeze=False):
        """
        Create an instance of LowerTriangular transform.
        Args:
            N the size of the final lower triangular matrices.
            num_matrices: Number of matrices to be stored.
            squeeze: If num_matrices == 1, drop the redundant axis.
        """
        self.num_matrices = num_matrices  # We need to store this for reconstruction.
        self.squeeze = squeeze
        self.N = N

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
        fwd = tf.transpose(tfw.vec_to_tri(tf.reshape(x, (self.num_matrices, -1)),self.N), [1, 2, 0])
        return tf.squeeze(fwd) if self.squeeze else fwd

    def tf_log_jacobian(self, x):
        return tf.zeros((1,), float_type)

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


positive = Log1pe()
