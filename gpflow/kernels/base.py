# Copyright 2018 GPflow
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
r"""
Kernels form a core component of GPflow models and allow prior information to
be encoded about a latent function of interest. The effect of choosing
different kernels, and how it is possible to combine multiple kernels is shown
in the `"Using kernels in GPflow" notebook <notebooks/kernels.html>`_.
"""


import abc
from functools import partial, reduce
from typing import Optional

import numpy as np
import tensorflow as tf


class Kernel(tf.Module):
    """
    The basic kernel class. Handles active dims.
    """

    def __init__(self, active_dims: slice = None, name: str = None):
        """
        :param active_dims: active dimensions, has the slice type.
        """
        super().__init__(name)
        if isinstance(active_dims, list):
            active_dims = np.array(active_dims)
        self._active_dims = active_dims

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = np.array(value, dtype=int)
        self._active_dims = value

    def on_separate_dims(self, other):
        """
        Checks if the dimensions, over which the kernels are specified, overlap.
        Returns True if they are defined on different/separate dimensions and False otherwise.
        """
        if isinstance(self.active_dims, slice) or isinstance(other.active_dims, slice):
            # Be very conservative for kernels defined over slices of dimensions
            return False

        if self.active_dims is None or other.active_dims:
            return False

        this_dims = tf.reshape(self.active_dims, (-1, 1))
        other_dims = tf.reshape(other.active_dims, (1, -1))
        return not np.any(tf.equal(this_dims, other_dims))

    def slice(self, X: tf.Tensor, Y: Optional[tf.Tensor] = None):
        """
        Slice the correct dimensions for use in the kernel, as indicated by `self.active_dims`.

        :param X: Input 1 [N, D].
        :param Y: Input 2 [M, D], can be None.
        :return: Sliced X, Y, [N, I], I - input dimension.
        """
        dims = self.active_dims
        if isinstance(dims, slice):
            X = X[..., dims]
            Y = Y[..., dims] if Y is not None else X
        elif dims is not None:
            # TODO(@awav): Convert when TF2.0 whill support proper slicing.
            X = tf.gather(X, dims, axis=-1)
            Y = tf.gather(Y, dims, axis=-1) if Y is not None else X
        return X, Y

    def slice_cov(self, cov: tf.Tensor) -> tf.Tensor:
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.

        :param cov: Tensor of covariance matrices, [N, D, D] or [N, D].
        :return: [N, I, I].
        """
        if cov.ndim == 2:
            cov = tf.linalg.diag(cov)

        dims = self.active_dims

        if isinstance(dims, slice):
            return cov[..., dims, dims]
        elif dims is not None:
            nlast = cov.shape[-1]
            ndims = len(dims)

            cov_shape = cov.shape
            cov_reshaped = tf.reshape(cov, [-1, nlast, nlast])
            gather1 = tf.gather(tf.transpose(cov_reshaped, [2, 1, 0]), dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [ndims, ndims]], 0))

        return cov

    @abc.abstractmethod
    def K(self, X, Y=None, presliced=False):
        pass

    @abc.abstractmethod
    def K_diag(self, X, presliced=False):
        pass

    def __call__(self, X, Y=None, full=True, presliced=False):
        if not full and Y is not None:
            raise ValueError("Ambiguous inputs: `diagonal` and `y` are not compatible.")
        if not full:
            return self.K_diag(X, presliced=presliced)
        return self.K(X, Y, presliced=presliced)

    def __add__(self, other):
        return Sum([self, other])

    def __mul__(self, other):
        return Product([self, other])


class Combination(Kernel):
    """
    Combine a list of kernels, e.g. by adding or multiplying (see inheriting
    classes).

    The names of the kernels to be combined are generated from their class
    names.
    """

    _reduction = None

    def __init__(self, kernels, name=None):
        super().__init__(name=name)

        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        # add kernels to a list, flattening out instances of this class therein
        kernels_list = []
        for k in kernels:
            if isinstance(k, self.__class__):
                kernels_list.extend(k.kernels)
            else:
                kernels_list.append(k)
        self.kernels = kernels_list

    @property
    def on_separate_dimensions(self):
        """
        Checks whether the kernels in the combination act on disjoint subsets
        of dimensions. Currently, it is hard to asses whether two slice objects
        will overlap, so this will always return False.

        :return: Boolean indicator.
        """
        if np.any([isinstance(k.active_dims, slice) for k in self.kernels]):
            # Be conservative in the case of a slice object
            return False
        else:
            dimlist = [k.active_dims for k in self.kernels]
            overlapping = False
            for i, dims_i in enumerate(dimlist):
                for dims_j in dimlist[i + 1:]:
                    print(f"dims_i = {type(dims_i)}")
                    if np.any(dims_i.reshape(-1, 1) == dims_j.reshape(1, -1)):
                        overlapping = True
            return not overlapping

    def K(self, X, X2=None, presliced=False):
        res = [k.K(X, X2, presliced=presliced) for k in self.kernels]
        return self._reduce(res)

    def K_diag(self, X, presliced=False):
        res = [k.K_diag(X, presliced=presliced) for k in self.kernels]
        return self._reduce(res)


class Sum(Combination):
    @property
    def _reduce(cls):
        return tf.add_n


class Product(Combination):
    @property
    def _reduce(cls):
        return partial(reduce, tf.multiply)
