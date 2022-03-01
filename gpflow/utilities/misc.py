# Copyright 2017-2021 The GPflow Contributors. All Rights Reserved.
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

from typing import Callable, Iterable, List, Optional, Union

import tensorflow as tf
import tensorflow_probability as tfp

from ..base import TensorData
from ..config import default_float, default_int
from .ops import cast

__all__ = [
    "to_default_float",
    "to_default_int",
    "set_trainable",
    "is_variable",
    "training_loop",
]


def to_default_int(x):
    return cast(x, dtype=default_int())


def to_default_float(x):
    return cast(x, dtype=default_float())


def set_trainable(model: Union[tf.Module, Iterable[tf.Module]], flag: bool) -> None:
    """
    Set trainable flag for all `tf.Variable`s and `gpflow.Parameter`s in a `tf.Module` or collection
    of `tf.Module`s.
    """
    modules = [model] if isinstance(model, tf.Module) else model

    for mod in modules:
        for variable in mod.variables:
            variable._trainable = flag


def is_variable(t: TensorData) -> bool:
    """
    Returns whether the `t` is a TensorFlow variable.
    """
    return isinstance(t, (tf.Variable, tfp.util.TransformedVariable))


def training_loop(
    closure: Callable[[], tf.Tensor],
    optimizer: Optional[tf.optimizers.Optimizer] = None,
    var_list: List[tf.Variable] = None,
    maxiter=1e3,
    compile=False,
):
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
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(var_list)
            loss = closure()
        grads = tape.gradient(loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

    if compile:
        optimization_step = tf.function(optimization_step)

    for _ in range(int(maxiter)):
        optimization_step()
