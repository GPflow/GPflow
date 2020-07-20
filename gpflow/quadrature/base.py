from typing import Callable, Union, Iterable

import abc
import tensorflow as tf

from ..base import TensorType


class GaussianQuadrature:
    """
    Abstract class implementing quadrature methods to compute Gaussian Expectations.  
    Inheriting classes must provide the method _build_X_W to create points and weights
    to be used for quadrature.
    """

    @abc.abstractmethod
    def _build_X_W(self, mean: TensorType, var: TensorType):
        raise NotImplementedError

    def __call__(self, fun, mean, var, *args, **kwargs):
        r"""
        Compute the Gaussian Expectation of a function f:

            X ~ N(mean, var)
            E[f(X)] = ∫f(x, *args, **kwargs)p(x)dx

        Using the formula:
            E[f(X)] = sum_{i=1}^{N_quad_points} f(x_i) * w_i

        where x_i, w_i must be provided by the inheriting class through self._build_X_W.
        The computations broadcast along batch-dimensions, represented by [b1, b2, ..., bX].

        :param fun: Callable or Iterable of Callables that operates elementwise, with
            signature f(X, *args, **kwargs). Moreover, if must satisfy the shape-mapping:
                X shape: [b1, b2, ..., bX, N_quad_points, d],
                    usually [N, N_quad_points, d]
                f(X) shape: [b1, b2, ...., bf, N_quad_points, d'],
                    usually [N, N_quad_points, 1] or [N, N_quad_points, d]
            In most cases, f should only operate over the last dimension of X
        :param mean: Array/Tensor with shape [b1, b2, ..., bX, d], usually [N, d],
            representing the mean of a d-Variate Gaussian distribution
        :param var: Array/Tensor with shape b1, b2, ..., bX, d], usually [N, d],
            representing the variance of a d-Variate Gaussian distribution
        :param *args: Passed to fun
        :param **kargs: Passed to fun
        :return: Array/Tensor with shape [b1, b2, ...., bf, N_quad_points, d'],
            usually [N, d] or [N, 1]
        """

        X, W = self._build_X_W(mean, var)
        if isinstance(fun, Iterable):
            return [tf.reduce_sum(f(X, *args, **kwargs) * W, axis=-2) for f in fun]
        return tf.reduce_sum(fun(X, *args, **kwargs) * W, axis=-2)

    def logspace(self, fun: Union[Callable, Iterable[Callable]], mean, var, *args, **kwargs):
        r"""
        Compute the Gaussian log-Expectation of a the exponential of a function f:

            X ~ N(mean, var)
            log E[exp[f(X)]] = log ∫exp[f(x, *args, **kwargs)]p(x)dx

        Using the formula:
            log E[exp[f(X)]] = log sum_{i=1}^{N_quad_points} exp[f(x_i) + log w_i]

        where x_i, w_i must be provided by the inheriting class through self._build_X_W.
        The computations broadcast along batch-dimensions, represented by [b1, b2, ..., bX].

        :param fun: Callable or Iterable of Callables that operates elementwise, with
            signature f(X, *args, **kwargs). Moreover, if must satisfy the shape-mapping:
                X shape: [b1, b2, ..., bX, N_quad_points, d],
                    usually [N, N_quad_points, d]
                f(X) shape: [b1, b2, ...., bf, N_quad_points, d'],
                    usually [N, N_quad_points, 1] or [N, N_quad_points, d]
            In most cases, f should only operate over the last dimension of X
        :param mean: Array/Tensor with shape [b1, b2, ..., bX, d], usually [N, d],
            representing the mean of a d-Variate Gaussian distribution
        :param var: Array/Tensor with shape b1, b2, ..., bX, d], usually [N, d],
            representing the variance of a d-Variate Gaussian distribution
        :param *args: Passed to fun
        :param **kargs: Passed to fun
        :return: Array/Tensor with shape [b1, b2, ...., bf, N_quad_points, d'],
            usually [N, d] or [N, 1]
        """

        X, W = self._build_X_W(mean, var)
        logW = tf.math.log(W)
        if isinstance(fun, Iterable):
            return [tf.reduce_logsumexp(f(X, *args, **kwargs) + logW, axis=-2) for f in fun]
        return tf.reduce_logsumexp(fun(X, *args, **kwargs) + logW, axis=-2)
