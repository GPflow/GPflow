# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Optional

import tensorflow as tf

from ..base import Parameter, TensorType
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities import positive
from .base import ActiveDims, Kernel


class Linear(Kernel):
    """
    The linear kernel. Functions drawn from a GP with this kernel are linear, i.e. f(x) = cx.
    The kernel equation is

        k(x, y) = σ²xy

    where σ² is the variance parameter.
    """

    @check_shapes(
        "variance: [broadcast n_active_dims]",
    )
    def __init__(
        self, variance: TensorType = 1.0, active_dims: Optional[ActiveDims] = None
    ) -> None:
        """
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())
        self._validate_ard_active_dims(self.variance)

    @property
    def ard(self) -> bool:
        """
        Whether ARD behaviour is active.
        """
        ndims: int = self.variance.shape.ndims
        return ndims > 0

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            return tf.matmul(X * self.variance, X, transpose_b=True)
        else:
            return tf.tensordot(X * self.variance, X2, [[-1], [-1]])

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.reduce_sum(tf.square(X) * self.variance, axis=-1)


class Polynomial(Linear):
    """
    The Polynomial kernel. Functions drawn from a GP with this kernel are
    polynomials of degree `d`. The kernel equation is

        k(x, y) = (σ²xy + γ)ᵈ

    where:
    σ² is the variance parameter,
    γ is the offset parameter,
    d is the degree parameter.
    """

    @check_shapes(
        "variance: [broadcast n_active_dims]",
    )
    def __init__(
        self,
        degree: TensorType = 3.0,
        variance: TensorType = 1.0,
        offset: TensorType = 1.0,
        active_dims: Optional[ActiveDims] = None,
    ) -> None:
        """
        :param degree: the degree of the polynomial
        :param variance: the (initial) value for the variance parameter(s),
            to induce ARD behaviour this must be initialised as an array the same
            length as the the number of active dimensions e.g. [1., 1., 1.]
        :param offset: the offset of the polynomial
        :param active_dims: a slice or list specifying which columns of X are used
        """
        super().__init__(variance, active_dims)
        self.degree = degree
        self.offset = Parameter(offset, transform=positive())

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        return (super().K(X, X2) + self.offset) ** self.degree

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return (super().K_diag(X) + self.offset) ** self.degree
