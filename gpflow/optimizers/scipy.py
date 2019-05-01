from typing import Callable, Iterator, List, Tuple, Union, Optional

import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import OptimizeResult

__all__ = ['Scipy']

Loss = tf.Tensor
Variables = List[tf.Variable]
Gradients = List[tf.Tensor]
StepCallback = Callable[[Loss, Variables, Gradients], None]
LossClosure = Callable[..., Tuple[tf.Tensor, Variables]]


class Scipy:
    def minimize(self,
                 closure: LossClosure,
                 variables: Variables,
                 step_callback: Optional[StepCallback] = None,
                 name: str = None,
                 **scipy_kwargs) -> OptimizeResult:
        """
        Minimize is a proxy method for `scipy.optimize.minimize` function.
        Args:
            closure: A closure that re-evaluates the model and returns the loss. The closure
                should clear the gradients, compute the loss and gradients.
            scipy_kwargs: Arguments passed to `scipy.optimize.minimize` method.
        Returns:
            The optimization result represented as a scipy ``OptimizeResult`` object.
            See `OptimizeResult` for a attributes description.
        """
        if not callable(closure):
            raise ValueError('Callable object expected.')
        initial_params = self.initial_parameters(variables)
        func = self.eval_func(closure, variables, step_callback)
        return scipy.optimize.minimize(func,
                                       initial_params,
                                       jac=True,
                                       **scipy_kwargs)

    @classmethod
    def initial_parameters(cls, variables):
        return cls.pack_tensors(variables)

    @classmethod
    def eval_func(cls,
                  closure: LossClosure,
                  variables: Variables,
                  step_callback: Optional[StepCallback] = None):
        def _eval(x):
            cls.unpack_tensors(variables, x)
            loss, grads = _compute_loss_and_gradients(closure, variables)
            if callable(step_callback):
                step_callback(loss, variables, grads)
            return loss.numpy(), cls.pack_tensors(grads)

        return _eval

    @staticmethod
    def pack_tensors(tensors: Iterator[tf.Tensor]) -> np.ndarray:
        flats = [tf.reshape(tensor, (-1, )) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector.numpy()

    @staticmethod
    def unpack_tensors(to_tensors: Iterator[tf.Tensor],
                       from_vector: np.ndarray):
        s = 0
        for tensor in to_tensors:
            shape = tensor.shape
            tensor_size = int(np.prod(shape))
            tensor_vector = from_vector[s:s + tensor_size]
            tensor_vector = tf.reshape(tensor_vector, shape)
            tensor.assign(tensor_vector)
            s += tensor_size


def _compute_loss_and_gradients(loss_cb: LossClosure, variables: Variables):
    with tf.GradientTape() as tape:
        loss = loss_cb()
        grads = tape.gradient(loss, variables)
    return loss, grads
