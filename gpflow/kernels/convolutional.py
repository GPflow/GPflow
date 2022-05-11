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

from typing import Optional, Sequence, cast

import numpy as np
import tensorflow as tf

from ..base import Parameter, TensorType
from ..config import default_float
from ..experimental.check_shapes import check_shape as cs
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..utilities import to_default_float
from .base import Kernel


class Convolutional(Kernel):
    r"""
    Plain convolutional kernel as described in :cite:t:`vdw2017convgp`. Defines
    a GP :math:`f()` that is constructed from a sum of responses of individual patches
    in an image:

    .. math::
       f(x) = \sum_p x^{[p]}

    where :math:`x^{[p]}` is the :math:`p`'th patch in the image.

    The key reference is :cite:t:`vdw2017convgp`.
    """

    @check_shapes(
        "image_shape: [2]",
        "patch_shape: [2]",
        "weights: [P]",
    )
    def __init__(
        self,
        base_kernel: Kernel,
        image_shape: Sequence[int],
        patch_shape: Sequence[int],
        weights: Optional[TensorType] = None,
        colour_channels: int = 1,
    ) -> None:
        super().__init__()
        self.image_shape = image_shape
        self.patch_shape = patch_shape
        self.base_kernel = base_kernel
        self.colour_channels = colour_channels
        self.weights = Parameter(
            np.ones(self.num_patches, dtype=default_float()) if weights is None else weights
        )

    @check_shapes(
        "X: [batch..., N, D]",
        "return: [batch..., N, P, S]",
    )
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour
        channels.

        :param X: Images.
        :return: Patches.
        """
        # Roll the colour channel to the front, so it appears to
        # `tf.extract_image_patches()` as separate images. Then extract patches
        # and reshape to have the first axis the same as the number of images.
        # The separate patches will then be in the second axis.
        batch = tf.shape(X)[:-2]
        N = tf.shape(X)[-2]
        flat_batch = tf.reduce_prod(batch)
        num_data = flat_batch * N
        X = cs(
            tf.transpose(tf.reshape(X, [num_data, -1, self.colour_channels]), [0, 2, 1]),
            "[num_data, C, W_x_H]",
        )
        X = cs(
            tf.reshape(X, [-1, self.image_shape[0], self.image_shape[1], 1], name="rX"),
            "[num_data_x_C, W, H, 1]",
        )
        patches = cs(
            tf.image.extract_patches(
                X,
                [1, self.patch_shape[0], self.patch_shape[1], 1],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
                "VALID",
            ),
            "[num_data_x_C, n_x_patches, n_y_patches, S]",
        )
        shp = tf.shape(patches)
        reshaped_patches = cs(
            tf.reshape(
                patches, tf.concat([batch, [N, self.colour_channels * shp[1] * shp[2], shp[3]]], 0)
            ),
            "[batch..., N, P, S]",
        )
        return to_default_float(reshaped_patches)

    @inherit_check_shapes
    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        Xp = cs(self.get_patches(X), "[batch..., N, P, S]")
        W2 = cs(self.weights[:, None] * self.weights[None, :], "[P, P]")

        rank = tf.rank(Xp) - 3
        batch = tf.shape(Xp)[:-3]
        N = tf.shape(Xp)[-3]
        P = tf.shape(Xp)[-2]
        S = tf.shape(Xp)[-1]
        ones = tf.ones((rank,), dtype=tf.int32)

        if X2 is None:
            Xp = cs(tf.reshape(Xp, tf.concat([batch, [N * P, S]], 0)), "[batch..., N_x_P, S]")
            bigK = cs(self.base_kernel.K(Xp), "[batch..., N_x_P, N_x_P]")
            bigK = cs(
                tf.reshape(bigK, tf.concat([batch, [N, P, N, P]], 0)), "[batch..., N, P, N, P]"
            )
            W2 = cs(tf.reshape(W2, tf.concat([ones, [1, P, 1, P]], 0)), "[..., 1, P, 1, P]")
            W2bigK = cs(bigK * W2, "[batch..., N, P, N, P]")
            return cs(
                tf.reduce_sum(W2bigK, [rank + 1, rank + 3]) / self.num_patches ** 2.0,
                "[batch..., N, N]",
            )

        else:
            Xp2 = Xp if X2 is None else cs(self.get_patches(X2), "[batch2..., N2, P, S]")
            rank2 = tf.rank(Xp2) - 3
            ones2 = tf.ones((rank2,), dtype=tf.int32)
            bigK = cs(self.base_kernel.K(Xp, Xp2), "[batch..., N, P, batch2..., N2, P]")
            W2 = cs(
                tf.reshape(W2, tf.concat([ones, [1, P], ones2, [1, P]], 0)),
                "[..., 1, P, ..., 1, P]",
            )
            W2bigK = cs(bigK * W2, "[batch..., N, P, batch2..., N2, P]")
            return cs(
                tf.reduce_sum(W2bigK, [rank + 1, rank + rank2 + 3]) / self.num_patches ** 2.0,
                "[batch..., N, batch2..., N2]",
            )

    @inherit_check_shapes
    def K_diag(self, X: TensorType) -> tf.Tensor:
        Xp = cs(self.get_patches(X), "[batch..., N, P, S]")

        rank = tf.rank(Xp) - 3
        P = tf.shape(Xp)[-2]
        ones = tf.ones((rank,), dtype=tf.int32)

        W2 = cs(self.weights[:, None] * self.weights[None, :], "[P, P]")
        W2 = cs(tf.reshape(W2, tf.concat([ones, [1, P, P]], 0)), "[..., 1, P, P]")

        bigK = cs(self.base_kernel.K(Xp), "[batch..., N, P, P]")

        return tf.reduce_sum(bigK * W2, [rank + 1, rank + 2]) / self.num_patches ** 2.0

    @property
    def patch_len(self) -> int:
        return cast(int, np.prod(self.patch_shape))

    @property
    def num_patches(self) -> int:
        return (
            (self.image_shape[0] - self.patch_shape[0] + 1)
            * (self.image_shape[1] - self.patch_shape[1] + 1)
            * self.colour_channels
        )
