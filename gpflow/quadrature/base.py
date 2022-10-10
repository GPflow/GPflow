# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

import abc
from typing import Any, Callable, Iterable, Tuple, Union

import tensorflow as tf

from ..base import TensorType
from ..experimental.check_shapes import check_shapes


class GaussianQuadrature:
    """
    Abstract class implementing quadrature methods to compute Gaussian Expectations.
    Inheriting classes must provide the method _build_X_W to create points and weights
    to be used for quadrature.
    """

    @abc.abstractmethod
    @check_shapes(
        "mean: [batch..., D]",
        "var: [batch..., D]",
        "return[0]: [N_quad_points, batch..., D]",
        "return[1]: [N_quad_points, broadcast batch..., 1]",
    )
    def _build_X_W(self, mean: TensorType, var: TensorType) -> Tuple[tf.Tensor, tf.Tensor]:
        raise NotImplementedError

    @check_shapes(
        "mean: [in_batch..., D]",
        "var: [in_batch..., D]",
        "return: [n_funs..., out_batch..., broadcast D]",
    )
    def __call__(
        self,
        fun: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
        mean: TensorType,
        var: TensorType,
        *args: Any,
        **kwargs: Any,
    ) -> tf.Tensor:
        r"""
        Compute the Gaussian Expectation of a function f::

            X ~ N(mean, var)
            E[f(X)] = ∫f(x, *args, **kwargs)p(x)dx

        Using the formula::

            E[f(X)] = sum_{i=1}^{N_quad_points} f(x_i) * w_i

        where x_i, w_i must be provided by the inheriting class through self._build_X_W.

        :param fun: Callable or Iterable of Callables that operates elementwise, with
            signature f(X, \*args, \*\*kwargs). Moreover, it must satisfy the shape-mapping::

                X shape: [N_quad_points, batch..., d].
                f(X) shape: [N_quad_points, batch..., broadcast d].

            In most cases, f should only operate over the last dimension of X
        :param mean: Array/Tensor representing the mean of a d-Variate Gaussian distribution
        :param var: Array/Tensor representing the variance of a d-Variate Gaussian distribution
        :param args: Passed to fun
        :param kargs: Passed to fun
        :return: Gaussian expectation of fun
        """

        X, W = self._build_X_W(mean, var)
        if isinstance(fun, Iterable):
            return [tf.reduce_sum(f(X, *args, **kwargs) * W, axis=0) for f in fun]
        return tf.reduce_sum(fun(X, *args, **kwargs) * W, axis=0)

    @check_shapes(
        "mean: [in_batch..., D]",
        "var: [in_batch..., D]",
        "return: [n_fun..., out_batch..., broadcast D]",
    )
    def logspace(
        self,
        fun: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
        mean: TensorType,
        var: TensorType,
        *args: Any,
        **kwargs: Any,
    ) -> tf.Tensor:
        r"""
        Compute the Gaussian log-Expectation of a the exponential of a function f::

            X ~ N(mean, var)
            log E[exp[f(X)]] = log ∫exp[f(x, *args, **kwargs)]p(x)dx

        Using the formula::

            log E[exp[f(X)]] = log sum_{i=1}^{N_quad_points} exp[f(x_i) + log w_i]

        where x_i, w_i must be provided by the inheriting class through self._build_X_W.
        The computations broadcast along batch-dimensions, represented by [batch...].

        :param fun: Callable or Iterable of Callables that operates elementwise, with
            signature f(X, \*args, \*\*kwargs). Moreover, it must satisfy the shape-mapping::

                X shape: [N_quad_points, batch..., d].
                f(X) shape: [N_quad_points, batch..., broadcast d].

            In most cases, f should only operate over the last dimension of X
        :param mean: Array/Tensor representing the mean of a d-Variate Gaussian distribution
        :param var: Array/Tensor representing the variance of a d-Variate Gaussian distribution
        :param args: Passed to fun
        :param kwargs: Passed to fun
        :return: Gaussian log-expectation of the exponential of a function f
        """

        X, W = self._build_X_W(mean, var)
        logW = tf.math.log(W)
        if isinstance(fun, Iterable):
            return [tf.reduce_logsumexp(f(X, *args, **kwargs) + logW, axis=0) for f in fun]
        return tf.reduce_logsumexp(fun(X, *args, **kwargs) + logW, axis=0)
