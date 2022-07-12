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

from typing import Optional, Sequence

import tensorflow as tf

from ..base import Parameter, TensorType
from ..experimental.check_shapes import check_shape as cs
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities import positive
from .base import Combination, Kernel


class ChangePoints(Combination):
    r"""
    The ChangePoints kernel defines a fixed number of change-points along a 1d
    input space where different kernels govern different parts of the space.

    The kernel is by multiplication and addition of the base kernels with
    sigmoid functions (σ). A single change-point kernel is defined as::

        K₁(x, x') * (1 - σ(x)) * (1 - σ(x')) + K₂(x, x') * σ(x) * σ(x')

    where K₁ is deactivated around the change-point and K₂ is activated. The
    single change-point version can be found in :cite:t:`lloyd2014`. Each sigmoid
    is a logistic function defined as::

        σ(x) = 1 / (1 + exp{-s(x - x₀)})

    parameterized by location "x₀" and steepness "s".

    The key reference is :cite:t:`lloyd2014`.
    """

    @check_shapes(
        "locations: [n_change_points]",
        "steepness: [broadcast n_change_points]",
    )
    def __init__(
        self,
        kernels: Sequence[Kernel],
        locations: TensorType,
        steepness: TensorType = 1.0,
        name: Optional[str] = None,
    ):
        """
        :param kernels: list of kernels defining the different regimes
        :param locations: list of change-point locations in the 1d input space
        :param steepness: the steepness parameter(s) of the sigmoids, this can be
            common between them or decoupled
        """
        if len(kernels) != len(locations) + 1:
            raise ValueError(
                "Number of kernels ({nk}) must be one more than the number of "
                "changepoint locations ({nl})".format(nk=len(kernels), nl=len(locations))
            )

        if isinstance(steepness, Sequence) and len(steepness) != len(locations):
            raise ValueError(
                "Dimension of steepness ({ns}) does not match number of changepoint "
                "locations ({nl})".format(ns=len(steepness), nl=len(locations))
            )

        super().__init__(kernels, name=name)

        self.locations = Parameter(locations)
        self.steepness = Parameter(steepness, transform=positive())

    def _set_kernels(self, kernels: Sequence[Kernel]) -> None:
        # it is not clear how to flatten out nested change-points
        self.kernels = list(kernels)

    @inherit_check_shapes
    def K(self, X: tf.Tensor, X2: Optional[tf.Tensor] = None) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `ChangePoints` kernel requires a 1D input space.")

        rank = tf.rank(X) - 2
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        Ncp = tf.shape(self.locations)[0]
        sig_X = cs(self._sigmoids(X), "[batch..., N, 1, Ncp]")

        if X2 is None:
            rank2 = 0
            batch2 = tf.constant([], dtype=tf.int32)
            N2 = N
            sig_X2 = sig_X
            sig_X = cs(
                tf.reshape(sig_X, tf.concat([batch, [N, 1, Ncp]], 0)), "[batch..., N, 1, Ncp]"
            )
            sig_X2 = cs(
                tf.reshape(sig_X2, tf.concat([batch, [1, N, Ncp]], 0)), "[batch..., 1, N, Ncp]"
            )
        else:
            rank2 = tf.rank(X2) - 2
            batch2 = tf.shape(X2)[:-2]
            N2 = tf.shape(X2)[-2]

            sig_X2 = cs(self._sigmoids(X2), "[batch2..., N2, 1, Ncp]")

            ones = tf.ones((rank,), dtype=tf.int32)
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            sig_X = cs(
                tf.reshape(sig_X, tf.concat([batch, [N], ones2, [1, Ncp]], 0)),
                "[batch..., N, ..., 1, Ncp]",
            )
            sig_X2 = cs(
                tf.reshape(sig_X2, tf.concat([ones, [1], batch2, [N2, Ncp]], 0)),
                "[..., 1, batch2..., N2, Ncp]",
            )

        # `starters` are the sigmoids going from 0 -> 1, whilst `stoppers` go
        # from 1 -> 0.
        starters = cs(sig_X * sig_X2, "[batch..., N, batch2..., N2, Ncp]")
        stoppers = cs((1 - sig_X) * (1 - sig_X2), "[batch..., N, batch2..., N2, Ncp]")

        # prepend `starters` with ones and append ones to `stoppers` since the
        # first kernel has no start and the last kernel has no end
        ones = tf.ones(tf.concat([batch, [N], batch2, [N2, 1]], 0), dtype=X.dtype)
        starters = cs(tf.concat([ones, starters], axis=-1), "[batch..., N, batch2..., N2, Nkern]")
        stoppers = cs(tf.concat([stoppers, ones], axis=-1), "[batch..., N, batch2..., N2, Nkern]")

        # now combine with the underlying kernels
        kernel_stack = cs(
            tf.stack([k(X, X2) for k in self.kernels], axis=-1),
            "[batch..., N, batch2..., N2, Nkern]",
        )
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=-1)

    @inherit_check_shapes
    def K_diag(self, X: tf.Tensor) -> tf.Tensor:
        cs(X, "[batch..., N, 1]  # The `ChangePoints` kernel requires a 1D input space.")

        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        Ncp = tf.shape(self.locations)[0]

        sig_X = cs(
            tf.reshape(self._sigmoids(X), tf.concat([batch, [N, Ncp]], 0)), "[batch..., N, Ncp]"
        )

        ones = tf.ones(tf.concat([batch, [N, 1]], 0), dtype=X.dtype)
        starters = cs(tf.concat([ones, sig_X * sig_X], axis=-1), "[batch..., N, Nkern]")
        stoppers = cs(tf.concat([(1 - sig_X) * (1 - sig_X), ones], axis=-1), "[batch..., N, Nkern]")

        kernel_stack = cs(
            tf.stack([k(X, full_cov=False) for k in self.kernels], axis=-1), "[batch..., N, Nkern]"
        )
        return tf.reduce_sum(kernel_stack * starters * stoppers, axis=-1)

    @check_shapes(
        "X: [batch...]",
        "return: [batch..., Ncp]",
    )
    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        locations = tf.sort(self.locations)  # ensure locations are ordered
        locations = tf.reshape(locations, (-1,))
        steepness = tf.reshape(self.steepness, (-1,))
        return tf.sigmoid(steepness * (X[..., None] - locations))
