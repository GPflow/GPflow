from typing import Callable, Iterator, List, Tuple, Union

import numpy as np
import scipy.optimize
import tensorflow as tf
from scipy.optimize import OptimizeResult

__all__ = ['ScipyOptimizer']


LossClosure = Callable[..., Tuple[tf.Tensor, List[tf.Tensor]]]


class ScipyOptimizer:
    def minimize(self,
                 closure: LossClosure,
                 variables: List[tf.Variable],
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
        func = self.eval_func(closure, variables)
        return scipy.optimize.minimize(func, initial_params, jac=True, **scipy_kwargs)

    @classmethod
    def initial_parameters(cls, variables):
        return cls.pack_tensors(variables)

    @classmethod
    def eval_func(cls,
                  closure: LossClosure,
                  variables: List[tf.Variables]):
        def _eval(x, *args):
            cls.unpack_tensors(variables, x)
            loss, grads = closure(*args)
            return loss, cls.pack_tensors(grads)
        return _eval

    @staticmethod
    def pack_tensors(tensors: Iterator[tf.Tensor]) -> np.ndarray:
        flats = [tf.reshape(tensor, (-1,)) for tensor in tensors]
        tensors_vector = tf.concat(flats, axis=0)
        return tensors_vector.numpy()

    @staticmethod
    def unpack_tensors(to_tensors: Iterator[tf.Tensor], from_vector: np.ndarray):
        s = 0
        for tensor in to_tensors:
            tensor_size = np.prod(tensor.shape)
            tensor_vector = from_vector[s: s + tensor_size]
            tensor.assign(tensor_vector)
            s += tensor_size
