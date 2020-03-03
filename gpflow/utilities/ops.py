import copy
from typing import List, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def eye(num: int, value: tf.Tensor, dtype: Optional[tf.DType] = None) -> tf.Tensor:
    if dtype is not None:
        value = tf.cast(value, dtype)
    return tf.linalg.diag(tf.fill([num], value))


def add_to_diagonal(to_tensor: tf.Tensor, value: tf.Tensor):
    diag = tf.linalg.diag_part(to_tensor)
    new_diag = diag + value
    return tf.linalg.set_diag(to_tensor, new_diag)


def leading_transpose(tensor: tf.Tensor, perm: List[Union[int, type(...)]], leading_dim: int = 0) -> tf.Tensor:
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
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1:]], 0)
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


def pca_reduce(X: tf.Tensor, Q: tf.Tensor) -> tf.Tensor:
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    if Q > X.shape[1]:  # pragma: no cover
        raise ValueError('Cannot have more latent dimensions than observed')
    if isinstance(X, tf.Tensor):
        X = X.numpy()
        # TODO why not use tf.linalg.eigh?
    evals, evecs = np.linalg.eigh(np.cov(X.T))
    W = evecs[:, -Q:]
    return (X - X.mean(0)).dot(W)


# def pca_reduce(data: tf.Tensor, latent_dim: tf.Tensor) -> tf.Tensor:
#     """
#     A helpful function for linearly reducing the dimensionality of the data X
#     to Q.
#     :param X: data array of size N (number of points) x D (dimensions)
#     :param Q: Number of latent dimensions, Q < D
#     :return: PCA projection array of size [N, Q].
#     """
#     assert latent_dim <= data.shape[1], 'Cannot have more latent dimensions than observed'
#     x_cov = tfp.stats.covariance(data)
#     evals, evecs = tf.linalg.eigh(x_cov)
#     W = evecs[:, -latent_dim:]
#     return (data - tf.reduce_mean(data, axis=0, keepdims=True)) @ W
