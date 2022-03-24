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

import warnings
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import OptimizeResult

from ..base import AnyNDArray
from ..monitor.base import Monitor

__all__ = ["Scipy"]

Variables = Iterable[tf.Variable]  # deprecated
StepCallback = Union[Callable[[int, Sequence[tf.Variable], Sequence[tf.Tensor]], None], Monitor]
LossClosure = Callable[[], tf.Tensor]


class Scipy:
    def minimize(
        self,
        closure: LossClosure,
        variables: Sequence[tf.Variable],
        method: Optional[str] = "L-BFGS-B",
        step_callback: Optional[StepCallback] = None,
        compile: bool = True,
        allow_unused_variables: bool = False,
        **scipy_kwargs: Any,
    ) -> OptimizeResult:
        """
        Minimize `closure`.

        Minimize is a wrapper around the `scipy.optimize.minimize` function handling the packing and
        unpacking of a list of shaped variables on the TensorFlow side vs. the flat numpy array
        required on the Scipy side.

        :param closure: A closure that re-evaluates the model, returning the loss to be minimized.
        :param variables: The list (tuple) of variables to be optimized
            (typically `model.trainable_variables`)
        :param method: The type of solver to use in SciPy. Defaults to "L-BFGS-B".
        :param step_callback: If not None, a callable that gets called once after each optimisation
            step. The callable is passed the arguments `step`, `variables`, and `values`. `step` is
            the optimisation step counter, `variables` is the list of trainable variables as above,
            and `values` is the corresponding list of tensors of matching shape that contains their
            value at this optimisation step.
        :param compile: If True, wraps the evaluation function (the passed `closure` as well as its
            gradient computation) inside a `tf.function()`, which will improve optimization speed in
            most cases.
        :param allow_unused_variables: Whether to allow variables that are not actually used in the
            closure.
        :param scipy_kwargs: Arguments passed through to `scipy.optimize.minimize`.
            Note that Scipy's minimize() takes a `callback` argument, but you probably want to use
            our wrapper and pass in `step_callback`.
        :returns:
            The optimization result represented as a Scipy ``OptimizeResult`` object.
            See the Scipy documentation for description of attributes.
        """
        if not callable(closure):
            raise TypeError(
                "The 'closure' argument is expected to be a callable object."
            )  # pragma: no cover
        variables = tuple(variables)
        if not all(isinstance(v, tf.Variable) for v in variables):
            raise TypeError(
                "The 'variables' argument is expected to only contain tf.Variable instances"
                " (use model.trainable_variables, not model.trainable_parameters)"
            )  # pragma: no cover
        initial_params = self.initial_parameters(variables)

        func = self.eval_func(
            closure, variables, compile=compile, allow_unused_variables=allow_unused_variables
        )
        if step_callback is not None:
            if "callback" in scipy_kwargs:
                raise ValueError("Callback passed both via `step_callback` and `callback`")

            callback = self.callback_func(variables, step_callback)
            scipy_kwargs.update(dict(callback=callback))

        return scipy.optimize.minimize(
            func, initial_params, jac=True, method=method, **scipy_kwargs
        )

    @classmethod
    def initial_parameters(cls, variables: Sequence[tf.Variable]) -> tf.Tensor:
        return cls.pack_tensors(variables)

    @classmethod
    def eval_func(
        cls,
        closure: LossClosure,
        variables: Sequence[tf.Variable],
        compile: bool = True,
        allow_unused_variables: bool = False,
    ) -> Callable[[AnyNDArray], Tuple[AnyNDArray, AnyNDArray]]:
        first_call = True

        def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            nonlocal first_call

            values = cls.unpack_tensors(variables, x)
            cls.assign_tensors(variables, values)

            if first_call:
                # Only check for unconnected gradients on the first function evaluation.
                loss, grads = _compute_loss_and_gradients(
                    closure, variables, tf.UnconnectedGradients.NONE
                )
                grads = cls._filter_unused_variables(variables, grads, allow_unused_variables)
                first_call = False
            else:
                loss, grads = _compute_loss_and_gradients(
                    closure, variables, tf.UnconnectedGradients.ZERO
                )

            return loss, cls.pack_tensors(grads)

        if compile:
            _tf_eval = tf.function(_tf_eval)

        def _eval(x: AnyNDArray) -> Tuple[AnyNDArray, AnyNDArray]:
            loss, grad = _tf_eval(tf.convert_to_tensor(x))
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval

    @staticmethod
    def _filter_unused_variables(
        variables: Sequence[tf.Variable], grads: Sequence[tf.Tensor], allow_unused_variables: bool
    ) -> Sequence[tf.Tensor]:
        filtered_grads = []
        unused_variables = []
        for i, grad in enumerate(grads):
            if grad is None:
                variable = variables[i]
                filtered_grads.append(tf.zeros_like(variable))
                unused_variables.append(variable.name)
            else:
                filtered_grads.append(grad)

        if unused_variables:
            msg = (
                "Some variables does not have a gradient, and appear unused in / not connected to"
                f" the loss closure: {unused_variables}."
            )
            if allow_unused_variables:
                warnings.warn(msg)
            else:
                raise ValueError(msg)

        return filtered_grads

    @classmethod
    def callback_func(
        cls, variables: Sequence[tf.Variable], step_callback: StepCallback
    ) -> Callable[[AnyNDArray], None]:
        step = 0  # type: int

        def _callback(x: AnyNDArray) -> None:
            nonlocal step

            if isinstance(step_callback, Monitor):
                step_callback(step)
            else:
                values = cls.unpack_tensors(variables, x)
                step_callback(step, variables, values)

            step += 1

        return _callback

    @staticmethod
    def pack_tensors(tensors: Sequence[Union[tf.Tensor, tf.Variable]]) -> tf.Tensor:
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector

    @staticmethod
    def unpack_tensors(
        to_tensors: Sequence[Union[tf.Tensor, tf.Variable]], from_vector: tf.Tensor
    ) -> List[tf.Tensor]:
        s = 0
        values = []
        for target_tensor in to_tensors:
            shape = tf.shape(target_tensor)
            dtype = target_tensor.dtype
            tensor_size = tf.reduce_prod(shape)
            tensor_vector = from_vector[s : s + tensor_size]
            tensor = tf.reshape(tf.cast(tensor_vector, dtype), shape)
            values.append(tensor)
            s += tensor_size
        return values

    @staticmethod
    def assign_tensors(to_tensors: Sequence[tf.Variable], values: Sequence[tf.Tensor]) -> None:
        if len(to_tensors) != len(values):
            raise ValueError("to_tensors and values should have same length")
        for target, value in zip(to_tensors, values):
            target.assign(value)


def _compute_loss_and_gradients(
    loss_closure: LossClosure,
    variables: Sequence[tf.Variable],
    unconnected_gradients: tf.UnconnectedGradients,
) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(variables)
        loss = loss_closure()
    grads = tape.gradient(loss, variables, unconnected_gradients=unconnected_gradients)
    return loss, grads
