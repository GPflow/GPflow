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
from typing import List, Optional, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

EllipsisType = type(...)


def cast(
    value: Union[tf.Tensor, np.ndarray], dtype: tf.DType, name: Optional[str] = None
) -> tf.Tensor:
    if not tf.is_tensor(value):
        # TODO(awav): Release TF2.2 resolves this issue
        # workaround for https://github.com/tensorflow/tensorflow/issues/35938
        return tf.convert_to_tensor(value, dtype, name=name)
    return tf.cast(value, dtype, name=name)


def eye(num: int, value: tf.Tensor, dtype: Optional[tf.DType] = None) -> tf.Tensor:
    if dtype is not None:
        value = cast(value, dtype)
    return tf.linalg.diag(tf.fill([num], value))


def leading_transpose(
    tensor: tf.Tensor, perm: List[Union[int, EllipsisType]], leading_dim: int = 0
) -> tf.Tensor:
    """
    Transposes tensors with leading dimensions. Leading dimensions in
    permutation list represented via ellipsis `...`.
    When leading dimensions are found, `transpose` method
    considers them as a single grouped element indexed by 0 in `perm` list. So, passing
    `perm=[-2, ..., -1]`, you assume that your input tensor has [..., A, B] shape,
    and you want to move leading dims between A and B dimensions.
    Dimension indices in permutation list can be negative or positive. Valid positive
    indices start from 1 up to the tensor rank, viewing leading dimensions `...` as zero
    index.
    Example:
        a = tf.random.normal((1, 2, 3, 4, 5, 6))
            # [..., A, B, C],
            # where A is 1st element,
            # B is 2nd element and
            # C is 3rd element in
            # permutation list,
            # leading dimensions are [1, 2, 3]
            # which are 0th element in permutation
            # list
        b = leading_transpose(a, [3, -3, ..., -2])  # [C, A, ..., B]
        sess.run(b).shape
        output> (6, 4, 1, 2, 3, 5)
    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises: ValueError when `...` cannot be found.
    """
    perm = copy.copy(perm)
    idx = perm.index(...)
    perm[idx] = leading_dim

    rank = tf.rank(tensor)
    perm_tf = perm % rank

    leading_dims = tf.range(rank - len(perm) + 1)
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1 :]], 0)
    return tf.transpose(tensor, perm)


def broadcasting_elementwise(op, a, b):
    """
    Apply binary operation `op` to every pair in tensors `a` and `b`.

    :param op: binary operator on tensors, e.g. tf.add, tf.substract
    :param a: tf.Tensor, shape [n_1, ..., n_a]
    :param b: tf.Tensor, shape [m_1, ..., m_b]
    :return: tf.Tensor, shape [n_1, ..., n_a, m_1, ..., m_b]
    """
    flatres = op(tf.reshape(a, [-1, 1]), tf.reshape(b, [1, -1]))
    return tf.reshape(flatres, tf.concat([tf.shape(a), tf.shape(b)], 0))


def square_distance(X, X2):
    """
    Returns ||X - X2ᵀ||²
    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.

    This function can deal with leading dimensions in X and X2.
    In the sample case, where X and X2 are both 2 dimensional,
    for example, X is [N, D] and X2 is [M, D], then a tensor of shape
    [N, M] is returned. If X is [N1, S1, D] and X2 is [N2, S2, D]
    then the output will be [N1, S1, N2, S2].
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


def difference_matrix(X, X2):
    """
    Returns (X - X2ᵀ)

    This function can deal with leading dimensions in X and X2.
    For example, If X has shape [M, D] and X2 has shape [N, D],
    the output will have shape [M, N, D]. If X has shape [I, J, M, D]
    and X2 has shape [K, L, N, D], the output will have shape
    [I, J, M, K, L, N, D].
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


def pca_reduce(X: tf.Tensor, latent_dim: tf.Tensor) -> tf.Tensor:
    """
    A helpful function for linearly reducing the dimensionality of the input
    points X to `latent_dim` dimensions.

    :param X: data array of size N (number of points) x D (dimensions)
    :param latent_dim: Number of latent dimensions Q < D
    :return: PCA projection array of size [N, Q].
    """
    if latent_dim > X.shape[1]:  # pragma: no cover
        raise ValueError("Cannot have more latent dimensions than observed")
    X_cov = tfp.stats.covariance(X)
    evals, evecs = tf.linalg.eigh(X_cov)
    W = evecs[:, -latent_dim:]
    return (X - tf.reduce_mean(X, axis=0, keepdims=True)) @ W
