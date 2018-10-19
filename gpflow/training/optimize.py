from contextlib import contextmanager
from typing import Callable, List, Optional

import tensorflow as tf

tfe = tf.contrib.eager


LossCallback = Callable[[], tf.Tensor]
StepCallback = Callable[[int, tf.Tensor, List[tf.Tensor]], None]


def optimize(loss_cb: LossCallback,
             optimizer: tf.train.Optimizer,
             variables: List[tfe.Variable],
             steps: int,
             step_cb: Optional[StepCallback] = None):
    for iteration in range(steps):
        with tf.GradientTape() as tape:
            loss = loss_cb()
        grads = tape.gradient(loss, variables)
        with unconstrain_variables(variables):
            optimizer.apply_gradients(zip(grads, variables))
        if callable(step_cb):
            step_cb(iteration, loss, grads)


@contextmanager
def unconstrain_variables(variables: List[tfe.Variable]):
    def switch(constrained: bool = False):
        for v in variables:
            v.is_constrained = constrained
    switch(False)
    try:
        yield
    finally:
        switch(True)
