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


class Static(Kernel):
    """
    Kernels who don't depend on the value of the inputs are 'Static'.  The only
    parameter is a variance, σ².
    """

    @check_shapes(
        "variance: []",
    )
    def __init__(
        self, variance: TensorType = 1.0, active_dims: Optional[ActiveDims] = None
    ) -> None:
        super().__init__(active_dims)
        self.variance = Parameter(variance, transform=positive())

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))


class White(Static):
    """
    The White kernel: this kernel produces 'white noise'. The kernel equation is

        k(x_n, x_m) = δ(n, m) σ²

    where:
    δ(.,.) is the Kronecker delta,
    σ²  is the variance parameter.
    """

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            d = tf.fill(tf.shape(X)[:-1], tf.squeeze(self.variance))
            return tf.linalg.diag(d)
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)
            return tf.zeros(shape, dtype=X.dtype)


class Constant(Static):
    """
    The Constant (aka Bias) kernel. Functions drawn from a GP with this kernel
    are constant, i.e. f(x) = c, with c ~ N(0, σ^2). The kernel equation is

        k(x, y) = σ²

    where:
    σ²  is the variance parameter.
    """

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        if X2 is None:
            shape = tf.concat(
                [
                    tf.shape(X)[:-2],
                    tf.reshape(tf.shape(X)[-2], [1]),
                    tf.reshape(tf.shape(X)[-2], [1]),
                ],
                axis=0,
            )
        else:
            shape = tf.concat([tf.shape(X)[:-1], tf.shape(X2)[:-1]], axis=0)

        return tf.fill(shape, tf.squeeze(self.variance))
