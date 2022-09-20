# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
"""
Throughout GPflow, by default, latent functions being modelled with Gaussian
processes are assumed to have zero mean, f ~ GP(0, k(x,x')).

In some cases we may wish to model only the deviation from a fixed function
with a Gaussian process.  For flexibility this fixed function could be both
input dependent and parameterised function, μ(x; θ),
with some unknown parameters θ, resulting in f ~ GP(μ(x;θ), k(x,x')).

The GPflow :class:`MeanFunction <gpflow.mean_functions.MeanFunction>` class
allows this to be done whilst additionally learning parameters of the
parametric function.
"""
from typing import Collection, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

from .base import Module, Parameter, TensorType
from .config import default_float, default_int
from .experimental.check_shapes import check_shape as cs
from .experimental.check_shapes import check_shapes, inherit_check_shapes


class Function(Module):
    """
    The base function class.
    To implement a function, write the ``__call__`` method. This takes a
    tensor X and returns a tensor f(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    :class:`Function` classes can have parameters, see the :class:`Linear` class for an
    example.
    """

    @check_shapes(
        "X: [batch..., D]",
        "return: [batch..., Q]",
    )
    def __call__(self, X: TensorType) -> tf.Tensor:
        raise NotImplementedError("Implement the __call__ method for this mean function")

    def __add__(self, other: "Function") -> "Function":
        return Additive(self, other)

    def __mul__(self, other: "Function") -> "Function":
        return Product(self, other)


class MeanFunction(Function):
    """
    Mixin for :class:`Function`\ s that are appropriate for use as mean functions.
    """


class Additive(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function) -> None:
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.multiply(self.prod_1(X), self.prod_2(X))


class Linear(MeanFunction, Function):
    """
    y_i = A x_i + b
    """

    @check_shapes(
        "A: [broadcast D, broadcast Q]",
        "b: [broadcast Q]",
    )
    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b


class Identity(Linear, Function):
    """
    y_i = x_i
    """

    # The many type-ignores in this class is because we replace a field in the super class with a
    # property, which mypy doesn't like.

    def __init__(self, input_dim: Optional[int] = None) -> None:
        Linear.__init__(self)
        self.input_dim = input_dim

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        return X

    @property
    def A(self) -> tf.Tensor:  # type: ignore[override]
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )
        return tf.eye(self.input_dim, dtype=default_float())

    @property
    def b(self) -> tf.Tensor:  # type: ignore[override]
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )

        return tf.zeros(self.input_dim, dtype=default_float())

    @A.setter  # type: ignore[attr-defined, no-redef]
    def A(self, A: tf.Tensor) -> None:
        pass

    @b.setter  # type: ignore[attr-defined, no-redef]
    def b(self, b: tf.Tensor) -> None:
        pass


class Constant(MeanFunction, Function):
    @check_shapes(
        "c: [broadcast Q]",
    )
    def __init__(self, c: TensorType = None) -> None:
        super().__init__()
        c = np.zeros(1) if c is None else c
        self.c = Parameter(c)

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        tile_shape = tf.concat(
            [tf.shape(X)[:-1], [1]],
            axis=0,
        )
        reshape_shape = tf.concat(
            [tf.ones(shape=(tf.rank(X) - 1), dtype=default_int()), [-1]],
            axis=0,
        )
        return tf.tile(tf.reshape(self.c, reshape_shape), tile_shape)


class Zero(Constant, Function):
    def __init__(self, output_dim: int = 1) -> None:
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=X.dtype)


class Polynomial(MeanFunction, Function):
    """
    A generic polynomial mean function.
    """

    @check_shapes("w: [broadcast output_dim, broadcast n_terms]")
    def __init__(
        self, degree: int, input_dim: int = 1, output_dim: int = 1, w: Optional[TensorType] = None
    ) -> None:
        """
        :param degree: The degree of the polynomial.
        :param input_dim: Number of inputs / variables this polynomial is defined over.
        :param output_dim: Number of outputs / polynomials.
        :param w: Initial weights of the terms of the polynomial. The inner dimension (``n_terms``)
            should correspond to the powers returned by ``compute_powers``.
        """
        powers = cs(tuple(self.compute_powers(degree, input_dim)), "[n_terms, input_dim]")
        if w is None:
            w = cs([1.0] + (len(powers) - 1) * [0.0], "[n_terms]")
        w_shape = (output_dim, len(powers))
        self.powers = tf.constant(powers, dtype=default_float())
        self.w = Parameter(tf.broadcast_to(w, w_shape))

    @staticmethod
    def compute_powers(degree: int, input_dim: int) -> Sequence[Tuple[int, ...]]:
        """
        Computes integer tuples corresponding to the powers to raise inputs to.

        Specifically this returns, in lexicographical order, all tuples where:

        * The tuple has length `input_dim`.
        * The values are non-negative integers.
        * The sum of the tuple is no greater than `degree`.

        For example::

            compute_powers(degree=2, input_dim=3)

        returns::

            (0, 0, 0)
            (0, 0, 1)
            (0, 0, 2)
            (0, 1, 0)
            (0, 1, 1)
            (0, 2, 0)
            (1, 0, 0)
            (1, 0, 1)
            (1, 1, 0)
            (2, 0, 0)

        where a tuple::

            (1, 0, 2)

        will translate to a the term::

            w[i] * (x[0]**1) * (x[1]**0) * (x[2]**2)
        """
        if not input_dim:
            return [()]
        result = []
        for i in range(degree + 1):
            for inner in Polynomial.compute_powers(degree - i, input_dim - 1):
                result.append((i,) + inner)
        return result

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        raised = cs(tf.pow(X[..., None, :], self.powers), "[batch..., n_terms, input_dim]")
        prod = cs(tf.math.reduce_prod(raised, axis=-1), "[batch..., n_terms]")
        return tf.einsum("...i,ji->...j", prod, self.w)


class SwitchedFunction(MeanFunction, Function):
    """
    This class enables to use different (independent) functions respective
    to the data 'label'.
    We assume the 'label' is stored in the extra column of X.
    """

    def __init__(self, function_list: Collection[Function]) -> None:
        super().__init__()
        self.functions = function_list

    @inherit_check_shapes
    def __call__(self, X: TensorType) -> tf.Tensor:
        ind = tf.gather(tf.transpose(X), tf.shape(X)[1] - 1)  # ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(
            tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1] - 1))
        )  # X = X[:,:-1]

        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, len(self.functions))
        # apply the likelihood-function to each section of the data
        results = [m(x) for x, m in zip(x_list, self.functions)]
        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.functions))
        return tf.dynamic_stitch(partitions, results)


class SwitchedMeanFunction(SwitchedFunction):
    """
    Renamed :class:`SwitchedFunction` for backwards compatibility.
    """

    def __init__(self, meanfunction_list: Collection[MeanFunction]) -> None:
        super().__init__(function_list=meanfunction_list)

    @property
    def meanfunctions(self) -> Collection[MeanFunction]:
        return self.functions

    @meanfunctions.setter
    def meanfunctions(self, value: Collection[MeanFunction]) -> None:
        self.function_list = value
