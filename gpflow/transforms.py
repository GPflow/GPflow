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
import itertools

from . import settings
from .misc import vec_to_tri
from .core.base import ITransform


class Transform(ITransform):  # pylint: disable=W0223
    def __call__(self, other_transform):
        """
        Calling a Transform with another Transform results in a Chain of both.
        The following are equivalent:
        >>> Chain(t1, t2)
        >>> t1(t2)
        """
        if not isinstance(other_transform, Transform):
            raise TypeError("transforms can only be chained with other transforms: "
                            "perhaps you want t.forward(x)")
        return Chain(self, other_transform)


class Identity(Transform):
    """
    The identity transform: y = x
    """
    def forward_tensor(self, x):
        return tf.identity(x)

    def backward_tensor(self, y):
        return tf.identity(y)

    def forward(self, x):
        return x

    def backward(self, y):
        return y

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return '(none)'


class Chain(Transform):
    """
    Chain two transformations together:
    .. math::
       y = t_1(t_2(x))
    where y is the natural parameter and x is the free state
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def forward_tensor(self, x):
        return self.t1.forward_tensor(self.t2.forward_tensor(x))

    def backward_tensor(self, y):
        return self.t2.backward_tensor(self.t1.backward_tensor(y))

    def forward(self, x):
        return self.t1.forward(self.t2.forward(x))

    def backward(self, y):
        return self.t2.backward(self.t1.backward(y))

    def log_jacobian_tensor(self, x):
        return self.t1.log_jacobian_tensor(self.t2.forward_tensor(x)) +\
               self.t2.log_jacobian_tensor(x)

    def __str__(self):
        return "{} {}".format(self.t1.__str__(), self.t2.__str__())


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

    def backward_tensor(self, y):
        return tf.log(y - self._lower)

    def forward(self, x):
        return np.exp(x) + self._lower

    def backward(self, y):
        return np.log(y - self._lower)

    def log_jacobian_tensor(self, x):
        return tf.reduce_sum(x)

    def __str__(self):
        return 'Exp'


class Log1pe(Transform):
    """
    A transform of the form
    .. math::

       y = \log(1 + \exp(x))

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

    def backward_tensor(self, y):
        ys = tf.maximum(y - self._lower, tf.as_dtype(settings.float_type).min)
        return ys + tf.log(-tf.expm1(-ys))

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
        ys = np.maximum(y - self._lower, np.finfo(settings.float_type).eps)
        return ys + np.log(-np.expm1(-ys))

    def __str__(self):
        return '+ve'


class Logistic(Transform):
    """
    The logistic transform, useful for keeping variables constrained between the limits a and b:
    .. math::

       y = a + (b-a) s(x)
       s(x) = 1 / (1 + \exp(-x))
    """
    def __init__(self, a=0., b=1.):
        if a >= b:
            raise ValueError("a must be smaller than b")
        self.a, self.b = float(a), float(b)

    def forward_tensor(self, x):
        ex = tf.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def forward(self, x):
        ex = np.exp(-x)
        return self.a + (self.b - self.a) / (1. + ex)

    def backward_tensor(self, y):
        return -tf.log((self.b - self.a) / (y - self.a) - 1.)

    def backward(self, y):
        return -np.log((self.b - self.a) / (y - self.a) - 1.)

    def log_jacobian_tensor(self, x):
        return tf.reduce_sum(x - 2. * tf.log(tf.exp(x) + 1.) + np.log(self.b - self.a))

    def __str__(self):
        return "[{}, {}]".format(self.a, self.b)


class Rescale(Transform):
    """
    A transform that can linearly rescale parameters:
    .. math::
       y = factor * x

    Use `Chain` to combine this with another transform such as Log1pe:
    `Chain(Rescale(), otherTransform())` results in
       y = factor * t(x)
    `Chain(otherTransform(), Rescale())` results in
       y = t(factor * x)

    This is useful for avoiding overly large or small scales in optimization/MCMC.

    If you want a transform for a positive quantity of a given scale, you want
    >>> Rescale(scale)(positive)
    """
    def __init__(self, factor=1.0):
        self.factor = float(factor)

    def forward_tensor(self, x):
        return x * self.factor

    def forward(self, x):
        return x * self.factor

    def backward_tensor(self, y):
        return y / self.factor

    def backward(self, y):
        return y / self.factor

    def log_jacobian_tensor(self, x):
        N = tf.cast(tf.reduce_prod(tf.shape(x)), dtype=settings.float_type)
        factor = tf.cast(self.factor, dtype=settings.float_type)
        log_factor = tf.log(factor)
        return N * log_factor

    def __str__(self):
        return "{}*".format(self.factor)


class DiagMatrix(Transform):
    """
    A transform to represent diagonal matrices.

    The output of this transform is a N x dim x dim array of diagonal matrices.
    The constructor argument `dim` specifies the size of the matrices.

    To make a constraint over positive-definite diagonal matrices, chain this
    transform with a positive transform. For example, to get posdef matrices of size 2x2:
        t = DiagMatrix(2)(positive)

    """

    def __init__(self, dim=1):
        self.dim = dim

    def forward(self, x):
        # create diagonal matrices
        m = np.zeros((x.size * self.dim)).reshape(-1, self.dim, self.dim)
        x = x.reshape(-1, self.dim)
        m[(np.s_[:],) + np.diag_indices(x.shape[1])] = x
        return m

    def backward(self, y):
        # Return diagonals of matrices
        if len(y.shape) not in (2, 3) or not (y.shape[-1] == y.shape[-2] == self.dim):
            raise ValueError("shape of input does not match this transform")
        return y.reshape((-1, self.dim, self.dim)).diagonal(offset=0, axis1=1, axis2=2).flatten()

    def backward_tensor(self, y):
        reshaped = tf.reshape(y, shape=(-1, self.dim, self.dim))
        return tf.reshape(tf.matrix_diag_part(reshaped), shape=[-1])

    def forward_tensor(self, x):
        # create diagonal matrices
        return tf.matrix_diag(tf.reshape(x, (-1, self.dim)))

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return 'DiagMatrix'


class LowerTriangular(Transform):
    """
    A transform of the form

       tri_mat = vec_to_tri(x)

    x is a free variable, y is always a list of lower triangular matrices sized
    (D x N x N).
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
        var = np.zeros((self.num_matrices, matsize, matsize), settings.float_type)
        for i in range(self.num_matrices):
            indices = np.tril_indices(matsize, 0)
            var[(np.zeros(len(indices[0])).astype(int) + i,) + indices] = xr[i, :]
        return var.squeeze(axis=0) if self.squeeze else var

    def backward(self, y):
        """
        Transforms from the variable to the free state.
        Args:
            y: Variable representation.

        Returns:
            Free state.
        """
        N = int(np.sqrt(y.size / self.num_matrices))
        reshaped = np.reshape(y, (self.num_matrices, N, N))
        # return reshaped[np.tril_indices(N, 0)].T
        return np.vstack([reshaped[i, :, :][np.tril_indices(N, 0)] for i in range(len(reshaped))])

    def forward_tensor(self, x):
        reshaped = tf.reshape(x, (self.num_matrices, -1))
        fwd = vec_to_tri(reshaped, self.N)
        return tf.squeeze(fwd, axis=0) if self.squeeze else fwd

    def backward_tensor(self, y):
        """
        CAVEAT: Requires defined shape and can't work with unknown shape.
        """
        size = np.prod(y.shape.as_list())
        N = int(np.sqrt(size / self.num_matrices))
        reshaped = tf.reshape(y, shape=(self.num_matrices, N, N))
        indices = np.array([np.hstack(x) for x in
                            itertools.product(np.arange(self.num_matrices), np.dstack(np.tril_indices(N))[0])])
        triangular = tf.reshape(tf.gather_nd(reshaped, indices), shape=[-1])
        return triangular[None, :]

    def log_jacobian_tensor(self, x):
        return tf.zeros((1,), settings.float_type)

    def __str__(self):
        return "LoTri->vec"


positive = Log1pe()


def positiveRescale(scale):
    """
    The appropriate joint transform for positive parameters of a given `scale`

    This is a convenient shorthand for

        constrained = scale * log(1 + exp(unconstrained))
    """
    return Rescale(scale)(positive)
