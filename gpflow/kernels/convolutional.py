import numpy as np
import tensorflow as tf

from .base import Kernel
from ..base import Parameter
from ..config import default_float


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

    def __init__(self, basekern, img_size, patch_size, weights=None, colour_channels=1):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.basekern = basekern
        self.colour_channels = colour_channels
        self.weights = Parameter(np.ones(self.num_patches, dtype=default_float()) if weights is None
                                 else weights)

    # @lru_cache() -- Can we do some kind of memoizing with TF2?
    def get_patches(self, X):
        """
        Extracts patches from the images X. Patches are extracted separately for each of the colour channels.
        :param X: (N x input_dim)
        :return: Patches (N, num_patches, patch_size)
        """
        # Roll the colour channel to the front, so it appears to `tf.extract_image_patches()` as separate images. Then
        # extract patches and reshape to have the first axis the same as the number of images. The separate patches will
        # then be in the second axis.
        castX = tf.transpose(
            tf.reshape(X, [tf.shape(X)[0], -1, self.colour_channels]),
            [0, 2, 1])
        patches = tf.image.extract_patches(
            tf.reshape(castX, [-1, self.img_size[0], self.img_size[1], 1], name="rX"),
            [1, self.patch_size[0], self.patch_size[1], 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1], "VALID")
        shp = tf.shape(patches)  # img x out_rows x out_cols
        return tf.cast(tf.reshape(patches,
                                  [tf.shape(X)[0], self.colour_channels * shp[1] * shp[2], shp[3]]),
                       default_float())

    def K(self, X, X2=None):
        Xp = self.get_patches(X)  # [N, P, patch_len]
        Xp2 = Xp if X2 is None else self.get_patches(X2)

        bigK = self.basekern.K(Xp, Xp2)  # [N, num_patches, N, num_patches]

        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        W2bigK = bigK * W2[None, :, None, :]
        return tf.reduce_sum(W2bigK, [1, 3]) / self.num_patches ** 2.0

    def K_diag(self, X):
        Xp = self.get_patches(X)  # N x num_patches x patch_dim
        W2 = self.weights[:, None] * self.weights[None, :]  # [P, P]
        bigK = self.basekern.K(Xp)  # [N, P, P]
        return tf.reduce_sum(bigK * W2[None, :, :], [1, 2]) / self.num_patches ** 2.0

    @property
    def patch_len(self):
        return np.prod(self.patch_size)

    @property
    def num_patches(self):
        return (self.img_size[0] - self.patch_size[0] + 1) * (
                self.img_size[1] - self.patch_size[1] + 1) * self.colour_channels
