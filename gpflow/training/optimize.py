from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf


InputData = Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
Variables = List[tf.Variable]

LossCallback = Callable[..., tf.Tensor]
StepCallback = Callable[[int, tf.Tensor, List[tf.Tensor]], None]

Optimizer = Union[tf.optimizers.Optimizer]


def create_iterator(*args, batch_size=None, buffer_size=1000, shuffle=True, repeat=True):
    """
    Args:
        *args: Arguments
        batch_size: Number of elements in a batch.
        buffer_size: Number of TODO.
        shuffle: TODO.
        repeat: TODO.
    Return:
        Creates iterator over input data.
    """
    ds = tf.data.Dataset.from_tensor_slices(args)
    if shuffle:
        ds = ds.shuffle(buffer_size)
    if batch_size is not None:
        ds = ds.batch(batch_size)
    if repeat:
        ds = ds.repeat()
    return ds.make_one_shot_iterator()


def loss_gradients(loss_cb: LossCallback, variables: Variables):
    with tf.GradientTape() as tape:
        loss = loss_cb()
    grads = tape.gradient(loss, variables)
    return loss, grads


def optimize(loss_cb: LossCallback,
             optimizer: tf.optimizers.Optimizer,
             variables: List[tf.Variable],
             steps: int,
             step_cb: Optional[StepCallback] = None):
    for iteration in range(steps):
        loss, grads = loss_gradients(loss_cb, variables)
        optimizer.apply_gradients(zip(grads, variables))
        if callable(step_cb):
            step_cb(iteration, loss, grads)