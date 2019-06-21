import copy
import functools
import operator
from typing import List, Optional, Union

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
        a = tf.random.normal((1, 2, 3, 4, 5, 6))
            # [..., A, B, C],
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


@tf.custom_gradient
def square_distance2(a: tf.Tensor, b: Optional[tf.Tensor] = None) -> tf.Tensor:
    """
    Returns ||a - bᵀ||²

    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    """
    if b is None:
        reduced_a2 = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)
        distance = -2 * tf.matmul(a, a, transpose_b=True)
        distance += reduced_a2 + tf.linalg.adjoint(reduced_a2)

        def grad_fn(grad_output):
            n = functools.reduce(operator.mul, a.shape[:-1])
            reduced_a = tf.reduce_sum(a, axis=tf.range(a.shape.ndims - 1), keepdims=True)
            print(grad_output)
            grad_a = (n * a - reduced_a) * grad_output * 4.0
            return grad_a

        return distance, grad_fn

    reduced_a2 = tf.reduce_sum(tf.square(a), axis=-1)
    reduced_b2 = tf.reduce_sum(tf.square(b), axis=-1)

    distance = -2 * tf.tensordot(a, b, [[-1], [-1]])
    distance += broadcasting_elementwise(tf.add, reduced_a2, reduced_b2)

    def grad_fn(grad_output):
        n = functools.reduce(operator.mul, a.shape[:-1])
        m = functools.reduce(operator.mul, b.shape[:-1])
        reduced_a = tf.reduce_sum(a, axis=tf.range(a.shape.ndims - 1), keepdims=True)
        reduced_b = tf.reduce_sum(b, axis=tf.range(b.shape.ndims - 1), keepdims=True)
        print(grad_output)
        grad_a = (m * a - reduced_b) * grad_output * 2.0
        grad_b = (n * b - reduced_a) * grad_output * 2.0
        return grad_a, grad_b

    return distance, grad_fn


def square_distance(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
    """
    Returns ||a - bᵀ||²

    Due to the implementation and floating-point imprecision, the
    result may actually be very slightly negative for entries very
    close to each other.
    """
    if b is None:
        reduced_a2 = tf.reduce_sum(tf.square(a), axis=-1, keepdims=True)
        distance = -2 * tf.matmul(a, a, transpose_b=True)
        distance += reduced_a2 + tf.linalg.adjoint(reduced_a2)
        return distance

    reduced_a2 = tf.reduce_sum(tf.square(a), axis=-1)
    reduced_b2 = tf.reduce_sum(tf.square(b), axis=-1)

    distance = -2 * tf.tensordot(a, b, [[-1], [-1]])
    distance += broadcasting_elementwise(tf.add, reduced_a2, reduced_b2)

    return distance