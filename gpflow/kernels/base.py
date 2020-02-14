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
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf

from ..base import Module

ActiveDims = Union[slice, list]

class Kernel(Module, metaclass=abc.ABCMeta):
    """
    The basic kernel class. Handles active dims.
    """

    def __init__(self,
                 active_dims: Optional[ActiveDims] = None,
                 name: Optional[str] = None):
        """
        :param active_dims: active dimensions, either a slice or list of
            indices into the columns of X.
        :param name: optional kernel name.
        """
        super().__init__(name=name)
        self._active_dims = self._normalize_active_dims(active_dims)

    @staticmethod
    def _normalize_active_dims(value):
        if value is None:
            value = slice(None, None, None)
        if not isinstance(value, slice):
            value = np.array(value, dtype=int)
        return value

    @property
    def active_dims(self):
        return self._active_dims

    @active_dims.setter
    def active_dims(self, value):
        self._active_dims = self._normalize_active_dims(value)

    def on_separate_dims(self, other):
        """
        Checks if the dimensions, over which the kernels are specified, overlap.
        Returns True if they are defined on different/separate dimensions and False otherwise.
        """
        if isinstance(self.active_dims, slice) or isinstance(
                other.active_dims, slice):
            # Be very conservative for kernels defined over slices of dimensions
            return False

        if self.active_dims is None or other.active_dims is None:
            return False

        this_dims = self.active_dims.reshape(-1, 1)
        other_dims = other.active_dims.reshape(1, -1)
        return not np.any(this_dims == other_dims)

    def slice(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None):
        """
        Slice the correct dimensions for use in the kernel, as indicated by `self.active_dims`.

        :param X: Input 1 [N, D].
        :param X2: Input 2 [M, D], can be None.
        :return: Sliced X, X2, [N, I], I - input dimension.
        """
        dims = self.active_dims
        if isinstance(dims, slice):
            X = X[..., dims]
            if X2 is not None:
                X2 = X2[..., dims]
        elif dims is not None:
            # TODO(@awav): Convert when TF2.0 will support proper slicing.
            X = tf.gather(X, dims, axis=-1)
            if X2 is not None:
                X2 = tf.gather(X2, dims, axis=-1)
        return X, X2

    def slice_cov(self, cov: tf.Tensor) -> tf.Tensor:
        """
        Slice the correct dimensions for use in the kernel, as indicated by
        `self.active_dims` for covariance matrices. This requires slicing the
        rows *and* columns. This will also turn flattened diagonal
        matrices into a tensor of full diagonal matrices.

        :param cov: Tensor of covariance matrices, [N, D, D] or [N, D].
        :return: [N, I, I].
        """
        if cov.shape.ndims == 2:
            cov = tf.linalg.diag(cov)

        dims = self.active_dims

        if isinstance(dims, slice):
            return cov[..., dims, dims]
        elif dims is not None:
            nlast = tf.shape(cov)[-1]
            ndims = len(dims)

            cov_shape = tf.shape(cov)
            cov_reshaped = tf.reshape(cov, [-1, nlast, nlast])
            gather1 = tf.gather(tf.transpose(cov_reshaped, [2, 1, 0]), dims)
            gather2 = tf.gather(tf.transpose(gather1, [1, 0, 2]), dims)
            cov = tf.reshape(tf.transpose(gather2, [2, 0, 1]),
                             tf.concat([cov_shape[:-2], [ndims, ndims]], 0))

        return cov

    def _validate_ard_active_dims(self, ard_parameter):
        """
        Validate that ARD parameter matches the number of active_dims (provided active_dims
        has been specified as an array).
        """
        if self.active_dims is None or isinstance(self.active_dims, slice):
            # Can only validate parameter if active_dims is an array
            return

        if ard_parameter.shape.rank > 0 and ard_parameter.shape[0] != len(self.active_dims):
            raise ValueError(f"Size of `active_dims` {self.active_dims} does not match "
                             f"size of ard parameter ({ard_parameter.shape[0]})")

    @abc.abstractmethod
    def K(self, X, X2=None):
        raise NotImplementedError

    @abc.abstractmethod
    def K_diag(self, X):
        raise NotImplementedError

    def __call__(self, X, X2=None, full=True, presliced=False):
        if (not full) and (X2 is not None):
            raise ValueError("Ambiguous inputs: `not full` and `X2` are not compatible.")

        if not presliced:
            X, X2 = self.slice(X, X2)

        if not full:
            assert X2 is None
            return self.K_diag(X)

        else:
            return self.K(X, X2)

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

    def __init__(self, kernels: List[Kernel], name: Optional[str] = None):
        super().__init__(name=name)

        if not all(isinstance(k, Kernel) for k in kernels):
            raise TypeError("can only combine Kernel instances")  # pragma: no cover

        self._set_kernels(kernels)

    def _set_kernels(self, kernels: List[Kernel]):
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


class ReducingCombination(Combination):
    def __call__(self, X, X2=None, full=True, presliced=False):
        return self._reduce(
            [k(X, X2, full=full, presliced=presliced) for k in self.kernels]
        )

    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        return self._reduce([k.K(X, X2) for k in self.kernels])

    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        return self._reduce([k.K_diag(X) for k in self.kernels])

    @property
    @abc.abstractmethod
    def _reduce(self):
        pass


class Sum(ReducingCombination):
    @property
    def _reduce(self):
        return tf.add_n


class Product(ReducingCombination):
    @property
    def _reduce(self):
        return partial(reduce, tf.multiply)
