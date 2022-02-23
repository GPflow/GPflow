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

import numpy as np
import tensorflow as tf

from ..base import Parameter, TensorType
from ..config import default_float
from ..utilities import to_default_float
from .base import Kernel


class Convolutional(Kernel):
    r"""
    Plain convolutional kernel as described in \citet{vdw2017convgp}. Defines
    a GP f( ) that is constructed from a sum of responses of individual patches
    in an image:
      f(x) = \sum_p x^{[p]}
    where x^{[p]} is the pth patch in the image.

    @incollection{vdw2017convgp,
      title = {Convolutional Gaussian Processes},
      author = {van der Wilk, Mark and Rasmussen, Carl Edward and Hensman, James},
      booktitle = {Advances in Neural Information Processing Systems 30},
      year = {2017},
      url = {http://papers.nips.cc/paper/6877-convolutional-gaussian-processes.pdf}
    }
    """

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

    # @lru_cache() -- Can we do some kind of memoizing with TF2?
    def get_patches(self, X: TensorType) -> tf.Tensor:
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_shape)
        """
        # Roll the colour channel to the front, so it appears to
        # `tf.extract_image_patches()` as separate images. Then extract patches
        # and reshape to have the first axis the same as the number of images.
        # The separate patches will then be in the second axis.
        num_data = tf.shape(X)[0]
        castX = tf.transpose(tf.reshape(X, [num_data, -1, self.colour_channels]), [0, 2, 1])
        patches = tf.image.extract_patches(
            tf.reshape(castX, [-1, self.image_shape[0], self.image_shape[1], 1], name="rX"),
            [1, self.patch_shape[0], self.patch_shape[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            "VALID",
        )
        shp = tf.shape(patches)  # img x out_rows x out_cols
        reshaped_patches = tf.reshape(
            patches, [num_data, self.colour_channels * shp[1] * shp[2], shp[3]]
        )
        return to_default_float(reshaped_patches)

    def K(self, X: TensorType, X2: Optional[TensorType] = None) -> tf.Tensor:
        Xp = self.get_patches(X)  # [N, P, patch_len]
        Xp2 = Xp if X2 is None else self.get_patches(X2)

        bigK = self.base_kernel.K(Xp, Xp2)  # [N, num_patches, N, num_patches]

        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        W2bigK = bigK * W2[None, :, None, :]
        return tf.reduce_sum(W2bigK, [1, 3]) / self.num_patches ** 2.0

    def K_diag(self, X: TensorType) -> tf.Tensor:
        Xp = self.get_patches(X)  # N x num_patches x patch_dim
        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        bigK = self.base_kernel.K(Xp)  # [N, P, P]
        return tf.reduce_sum(bigK * W2[None, :, :], [1, 2]) / self.num_patches ** 2.0

    @property
    def patch_len(self) -> np.ndarray:
        return np.prod(self.patch_shape)

    @property
    def num_patches(self) -> int:
        return (
            (self.image_shape[0] - self.patch_shape[0] + 1)
            * (self.image_shape[1] - self.patch_shape[1] + 1)
            * self.colour_channels
        )
