import copy
import logging
from typing import List, Union, Callable

import numpy as np
import tensorflow as tf

NoneType = type(None)


def create_logger(name=None):
    return logging.getLogger('Temporary Logger Solution')


def default_jitter_eye(num_rows: int, num_columns: int = None, value: float = None) -> float:
    value = default_jitter() if value is None else value
    num_rows = int(num_rows)
    num_columns = int(num_columns) if num_columns is not None else num_columns
    return tf.eye(num_rows, num_columns=num_columns, dtype=default_float()) * value


def default_jitter() -> float:
    return 1e-6


def default_float() -> float:
    return np.float64


def default_int() -> int:
    return np.int32


def leading_transpose(tensor: tf.Tensor, perm: List[Union[int, type(...)]]) -> tf.Tensor:
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
        b = leading_transpose(a, [5, -3, ..., -2])
        sess.run(b).shape
        output> (6, 4, 1, 2, 3, 5)
    :param tensor: TensorFlow tensor.
    :param perm: List of permutation indices.
    :returns: TensorFlow tensor.
    :raises: ValueError when `...` cannot be found.
    """
    perm = copy.copy(perm)
    idx = perm.index(...)
    perm[idx] = 0

    rank = tf.rank(tensor)
    perm_tf = perm % rank

    leading_dims = tf.range(rank - len(perm) + 1)
    perm = tf.concat([perm_tf[:idx], leading_dims, perm_tf[idx + 1:]], 0)
    return tf.transpose(tensor, perm)


def set_trainable(model: tf.Module, flag: bool = False):
    for variable in model.trainable_variables:
        variable._trainable = flag


def training_loop(closure: Callable[..., tf.Tensor],
                  optimizer=tf.optimizers.Adam(),
                  var_list: List[tf.Variable] = None,
                  apply_jit=True,
                  maxiter=1e3):
    """
    Simple generic training loop in TF-2.0. At each iteration uses a GradientTape to compute
    the gradients of a loss function with respect to a set of variables.
    :param closure: Callable that constructs a loss function based on data and model being trained
    :param optimizer: tf.optimizers or tf.keras.optimizers that updates variables by applying the
    corresponding loss gradients
    :param var_list: List of model variables to be learnt during training
    :param maxiter: Maximum number of
    :return:
    """

    def optimization_step():
        with tf.GradientTape() as tape:
            tape.watch(var_list)
            loss = closure()
            grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if apply_jit:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()
