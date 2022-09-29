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

import copy
from typing import Any, Callable, List, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from ..base import AnyNDArray
from ..experimental.check_shapes import check_shapes


@check_shapes(
    "value: []",
    "return: [N, N]",
)
def eye(num: int, value: tf.Tensor, dtype: Optional[tf.DType] = None) -> tf.Tensor:
    if dtype is not None:
        value = tf.cast(value, dtype)
    return tf.linalg.diag(tf.fill([num], value))


@check_shapes(
    "tensor: [any...]",
    "return: [transposed_any...]",
)
def leading_transpose(tensor: tf.Tensor, perm: List[Any], leading_dim: int = 0) -> tf.Tensor:
    """
    Transposes tensors with leading dimensions.

    Leading dimensions in permutation list represented via ellipsis `...` and is of type
    List[Union[int, type(...)]  (please note, due to mypy issues, List[Any] is used instead).  When
    leading dimensions are found, `transpose` method considers them as a single grouped element
    indexed by 0 in `perm` list. So, passing `perm=[-2, ..., -1]`, you assume that your input tensor
    has [..., A, B] shape, and you want to move leading dims between A and B dimensions.  Dimension
    indices in permutation list can be negative or positive. Valid positive indices start from 1 up
    to the tensor rank, viewing leading dimensions `...` as zero index.

    Example::

        a = tf.random.normal((1, 2, 3, 4, 5, 6))
        # [..., A, B, C],
        # where A is 1st element,
        # B is 2nd element and
        # C is 3rd element in
        # permutation list,
        # leading dimensions are [1, 2, 3]
        # which are 0th element in permutation list
        b = leading_transpose(a, [3, -3, ..., -2])  # [C, A, ..., B]
        sess.run(b).shape

        output> (6, 4, 1, 2, 3, 5)

    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises ValueError: when `...` cannot be found.

    """
    perm = copy.copy(perm)
    idx = perm.index(...)
    perm[idx] = leading_dim

    rank = tf.rank(tensor)
    perm_tf = perm % rank

    leading_dims = tf.range(rank - len(perm) + 1)
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1 :]], 0)
    return tf.transpose(tensor, perm)


@check_shapes(
    "a: [a_shape...]",
    "b: [b_shape...]",
    "return: [a_shape..., b_shape...]",
)
def broadcasting_elementwise(
    op: Callable[[tf.Tensor, tf.Tensor], tf.Tensor], a: tf.Tensor, b: tf.Tensor
) -> tf.Tensor:
    """
    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    """
    flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


@check_shapes(
    "X: [batch..., N, D]",
    "X2: [batch2..., N2, D]",
    "return: [batch..., N, batch2..., N2] if X2 is not None",
    "return: [batch..., N, N] if X2 is None",
)
def square_distance(X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    """
    Returns ||X - X2ᵀ||²

    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    """
    if X2 is None:
        Xs = tf.reduce_sum(tf.square(X), axis=-1, keepdims=True)
        dist = -2 * tf.matmul(X, X, transpose_b=True)
        dist += Xs + tf.linalg.adjoint(Xs)
        return dist
    Xs = tf.reduce_sum(tf.square(X), axis=-1)
    X2s = tf.reduce_sum(tf.square(X2), axis=-1)
    dist = -2 * tf.tensordot(X, X2, [[-1], [-1]])
    dist += broadcasting_elementwise(tf.add, Xs, X2s)
    return dist


@check_shapes(
    "X: [batch..., N, D]",
    "X2: [batch2..., N2, D]",
    "return: [batch..., N, batch2..., N2, D] if X2 is not None",
    "return: [batch..., N, N, D] if X2 is None",
)
def difference_matrix(X: tf.Tensor, X2: Optional[tf.Tensor]) -> tf.Tensor:
    """
    Returns (X - X2ᵀ)
    """
    if X2 is None:
        X2 = X
        diff = X[..., :, tf.newaxis, :] - X2[..., tf.newaxis, :, :]
        return diff
    Xshape = tf.shape(X)
    X2shape = tf.shape(X2)
    X = tf.reshape(X, (-1, Xshape[-1]))
    X2 = tf.reshape(X2, (-1, X2shape[-1]))
    diff = X[:, tf.newaxis, :] - X2[tf.newaxis, :, :]
    diff = tf.reshape(diff, tf.concat((Xshape[:-1], X2shape[:-1], [Xshape[-1]]), 0))
    return diff


@check_shapes(
    "X: [N, D]",
    "latent_dim: []",
    "return: [N, Q]",
)
def pca_reduce(X: tf.Tensor, latent_dim: tf.Tensor) -> tf.Tensor:
    """
    Linearly reduce the dimensionality of the input points `X` to `latent_dim` dimensions.

    :param X: Data to reduce.
    :param latent_dim: Number of latent dimension, Q < D.
    :return: PCA projection array.

    """
    if latent_dim > X.shape[1]:  # pragma: no cover
        raise ValueError("Cannot have more latent dimensions than observed")
    X_cov = tfp.stats.covariance(X)
    evals, evecs = tf.linalg.eigh(X_cov)
    W = evecs[:, -latent_dim:]
    return (X - tf.reduce_mean(X, axis=0, keepdims=True)) @ W
