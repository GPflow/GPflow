from typing import Callable, Iterator, List, Tuple, Union, Optional

import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import OptimizeResult

__all__ = ['Scipy']

Loss = tf.Tensor
Variables = Tuple[tf.Variable]
StepCallback = Callable[[int, Variables, List[tf.Tensor]], None]
LossClosure = Callable[..., Tuple[tf.Tensor, Variables]]


class Scipy:
    def minimize(self,
                 closure: LossClosure,
                 variables: Variables,
                 method: Optional[str] = "L-BFGS-B",
                 step_callback: Optional[StepCallback] = None,
                 jit: bool = True,
                 **scipy_kwargs) -> OptimizeResult:
        """
        Minimize is a wrapper around the `scipy.optimize.minimize` function
        handling the packing and unpacking of a list of shaped variables on the
        TensorFlow side vs. the flat numpy array required on the Scipy side.

        Args:
            closure: A closure that re-evaluates the model, returning the loss
                to be minimized.
            variables: The list (tuple) of variables to be optimized
                (typically `model.trainable_variables`)
            step_callback: If not None, a callable that gets called once after
                each optimisation step. The callabe is passed the arguments
                `step`, `variables`, and `values`. `step` is the optimisation
                step counter. `variables` is the list of trainable variables as
                above, and `values` is the corresponding list of tensors of
                matching shape that contains their value at this optimisation
                step.
            jit: If True, wraps the evaluation function (the passed `closure` as
                well as its gradient computation) inside a `tf.function()`,
                which will improve optimization speed in most cases.

            scipy_kwargs: Arguments passed through to `scipy.optimize.minimize`

        Returns:
            The optimization result represented as a scipy ``OptimizeResult``
            object. See the Scipy documentation for description of attributes.
        """
        if not callable(closure):
            raise TypeError('Callable object expected.')  # pragma: no cover
        initial_params = self.initial_parameters(variables)

        func = self.eval_func(closure, variables, jit=jit)
        if step_callback is not None:
            if 'callback' in scipy_kwargs:
                raise ValueError("Callback passed both via `step_callback` and `callback`")

            callback = self.callback_func(variables, step_callback)
            scipy_kwargs.update(dict(callback=callback))

        return scipy.optimize.minimize(func, initial_params, jac=True, method=method,
                                       **scipy_kwargs)

    @classmethod
    def initial_parameters(cls, variables):
        return cls.pack_tensors(variables)

    @classmethod
    def eval_func(cls, closure: LossClosure, variables: Variables, jit: bool = True):
        def _tf_eval(x: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
            values = cls.unpack_tensors(variables, x)
            cls.assign_tensors(variables, values)

            loss, grads = _compute_loss_and_gradients(closure, variables)
            return loss, cls.pack_tensors(grads)

        if jit:
            _tf_eval = tf.function(_tf_eval)

        def _eval(x):
            loss, grad = _tf_eval(tf.convert_to_tensor(x))
            return loss.numpy().astype(np.float64), grad.numpy().astype(np.float64)

        return _eval

    @classmethod
    def callback_func(cls, variables: Variables, step_callback: StepCallback):
        step = 0  # type: int

        def _callback(x):
            nonlocal step
            values = cls.unpack_tensors(variables, x)
            step_callback(step=step, variables=variables, values=values)
            step += 1

        return _callback

    @staticmethod
    def pack_tensors(tensors: Iterator[tf.Tensor]) -> tf.Tensor:
        flats = [tf.reshape(tensor, (-1, )) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector

    @staticmethod
    def unpack_tensors(to_tensors: Iterator[tf.Tensor], from_vector: tf.Tensor) -> List[tf.Tensor]:
        s = 0
        values = []
        for tensor in to_tensors:
            shape = tf.shape(tensor)
            tensor_size = tf.reduce_prod(shape)
            tensor_vector = tf.cast(from_vector[s:s + tensor_size], tensor.dtype)
            tensor_vector = tf.reshape(tensor_vector, shape)
            values.append(tensor_vector)
            s += tensor_size
        return values

    @staticmethod
    def assign_tensors(to_tensors: Iterator[tf.Variable], values: Iterator[tf.Tensor]):
        for tensor, tensor_vector in zip(to_tensors, values):
            tensor.assign(tensor_vector)


def _compute_loss_and_gradients(loss_cb: LossClosure, variables: Variables):
    with tf.GradientTape() as tape:
        loss = loss_cb()
    grads = tape.gradient(loss, variables)
    return loss, grads
