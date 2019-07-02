from typing import Callable, List, Optional, Dict

import tensorflow as tf

from .printing import leaf_components


__all__ = [
    "set_trainable",
    "multiple_assign",
    "training_loop"
]


def set_trainable(model: tf.Module, flag: bool = False):
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a module.
    """
    for variable in model.trainable_variables:
        variable._trainable = flag


def multiple_assign(input: tf.Module, vars_dict: Dict[str, tf.Tensor]):
    """
    Multiple assign takes a dictionary with new values. Dictionary keys are paths to the
    `tf.Variable`s or `gpflow.Parameters` of the input module.

    :param input: `tf.Module`.
    :param vars_dict: a dictionary with keys of the form "module.path.to.variable" and new value tensors.
    """
    reference_var_dict = leaf_components(input)
    for path, value in vars_dict.items():
        reference_var_dict[path].assign(value)


def training_loop(closure: Callable[..., tf.Tensor],
                  optimizer: Optional[tf.optimizers.Optimizer] = None,
                  var_list: List[tf.Variable] = None,
                  maxiter=1e3,
                  jit=False):
    """
    Simple generic training loop. At each iteration uses a GradientTape to compute
    the gradients of a loss function with respect to a set of variables.

    :param closure: Callable that constructs a loss function based on data and model being trained
    :param optimizer: tf.optimizers or tf.keras.optimizers that updates variables by applying the
        corresponding loss gradients. Adam is a default optimizer with default settings.
    :param var_list: List of model variables to be learnt during training
    :param maxiter: Maximum number of
    :return:
    """

    optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

    def optimization_step():
        with tf.GradientTape() as tape:
            tape.watch(var_list)
            loss = closure()
            grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if jit:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()
