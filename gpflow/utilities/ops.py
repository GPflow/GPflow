import copy
from typing import List, Union, Optional

import tensorflow as tf


def eye(num: int, value: tf.Tensor, dtype: Optional[tf.DType] = None) -> tf.Tensor:
    if dtype is not None:
        value = tf.cast(value, dtype)
    return tf.linalg.diag(tf.fill([num], value))


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
        a = tf.random.normal((1, 2, 3, 4, 5, 6))  # [..., A, B, C],
                                                  # where A is 1st element,
                                                  # B is 2nd element and
                                                  # C is 3rd element in
                                                  # permutation list,
                                                  # leading dimentions are [1, 2, 3]
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
    Returns (X - X2ᵀ)²
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
