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

from . import settings
from .misc import vec_to_tri
from .core.base import ITransform


class Transform(ITransform): # pylint: disable=W0223
    pass


class Identity(Transform):
    """
    The identity transform: y = x
    """
    def forward_tensor(self, x):
        return tf.identity(x)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.tf_float)

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

    def forward_tensor(self, x):
        return tf.exp(x) + self._lower

    def forward(self, x):
        return np.exp(x) + self._lower

    def backward(self, y):
        return np.log(y - self._lower)

    def log_jacobian_tensor(self, x):
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

    def forward_tensor(self, x):
        return tf.nn.softplus(x) + self._lower

    def log_jacobian_tensor(self, x):
        return tf.negative(tf.reduce_sum(tf.nn.softplus(tf.negative(x))))

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
        ys = np.maximum(y - self._lower, np.finfo(settings.np_float).eps)
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

    def forward_tensor(self, x):
        ex = tf.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def log_jacobian_tensor(self, x):
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

    def forward_tensor(self, x):
        return self.chain_transform.forward_tensor(x * self.factor)

    def forward(self, x):
        return self.chain_transform.forward(x * self.factor)

    def backward(self, y):
        return self.chain_transform.backward(y) / self.factor

    def log_jacobian_tensor(self, x):
        N = tf.cast(tf.reduce_prod(tf.shape(x)), dtype=settings.tf_float)
        factor = tf.cast(self.factor, dtype=settings.tf_float)
        log_factor = tf.log(factor)
        log_jacobian = self.chain_transform.log_jacobian_tensor(x * self.factor)
        return N * log_factor + log_jacobian

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

    def forward_tensor(self, x):
        return tf.matrix_diag(tf.reshape(self._positive_transform.forward_tensor(x), (-1, self.dim)))

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.tf_float) + self._positive_transform.log_jacobian_tensor(x)

    def __str__(self):
        return 'DiagMatrix'


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
        var = np.zeros((matsize, matsize, self.num_matrices), settings.np_float)
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
        N = int(np.sqrt(y.size / self.num_matrices))
        reshaped = np.reshape(y, (N, N, self.num_matrices))
        size = len(reshaped)
        triangular = reshaped[np.tril_indices(size, 0)].T
        return triangular

    def forward_tensor(self, x):
        reshaped = tf.reshape(x, (self.num_matrices, -1))
        fwd = tf.transpose(vec_to_tri(reshaped, self.N), [1, 2, 0])
        return tf.squeeze(fwd) if self.squeeze else fwd

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.tf_float)

    def __str__(self):
        return "LoTri->vec"


positive = Log1pe()
