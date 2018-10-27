from .scipy import ScipyOptimizer
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple, Union

import tensorflow as tf

tfe = tf.contrib.eager


InputData = Union[Tuple[tf.Tensor, tf.Tensor], tf.Tensor]
Variables = List[tf.Variable]

LossCallback = Callable[..., tf.Tensor]
StepCallback = Callable[[int, tf.Tensor, List[tf.Tensor]], None]

Optimizer = Union[tf.train.Optimizer, ]


def closure_func(loss_cb: LossCallback, data: InputData, variables: Variables):
    with tf.GradientTape() as tape:
        loss = loss_cb(*data)
    grads = tape.gradient(loss, variables)
    return loss, grads


def optimize(loss_cb: LossCallback,
             optimizer: tf.train.Optimizer,
             variables: List[tfe.Variable],
             steps: int,
             step_cb: Optional[StepCallback] = None):
    for iteration in range(steps):
        loss, grads = closure_func(loss_cb, [], variables)
        optimizer.apply_gradients(zip(grads, variables))
        if callable(step_cb):
            step_cb(iteration, loss, grads)


# @contextmanager
# def unconstrain_variables(variables: List[tfe.Variable]):
#     def switch(constrained: bool = False):
#         for v in variables:
#             v.is_constrained = constrained
#     switch(False)
#     try:
#         yield
#     finally:
#         switch(True)
