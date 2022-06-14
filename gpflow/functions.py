# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Callable, Collection, Iterator, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from .base import Module, Parameter, TensorData, TensorType
from .config import default_float, default_int
from .experimental.check_shapes import check_shape as cs
from .experimental.check_shapes import check_shapes
from .utilities import positive


class Function(Module):
    """
    The base function class.
    To implement a function, write the __call__ method. This takes a
    tensor X and returns a tensor f(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    :class:`Function` classes can have parameters, see the :class:`Linear` class for an
    example.
    """

    def __call__(self, X: TensorType) -> tf.Tensor:
        raise NotImplementedError("Implement the __call__ method for this mean function")

    def __add__(self, other: "Function") -> "Function":
        return Additive(self, other)

    def __mul__(self, other: "Function") -> "Function":
        return Product(self, other)


class MeanFunction(Function):
    """
    Mixin for :class:`Function`s that are appropriate for use as mean functions.
    """


class Additive(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function) -> None:
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction, Function):
    def __init__(self, first_part: Function, second_part: Function):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X: TensorType) -> tf.Tensor:
        return tf.multiply(self.prod_1(X), self.prod_2(X))


ConstantOrFunction = Union[Function, TensorData]
ParameterOrFunction = Union[Function, Parameter]


def prepare_constant_or_function(
    value: ConstantOrFunction,
    *,
    lower_bound: Optional[float] = None,
) -> ParameterOrFunction:
    if isinstance(value, Function):
        return value
    else:
        if lower_bound is None:
            return Parameter(value)
        else:
            return Parameter(value, transform=positive(lower_bound))


@check_shapes(
    "X: [batch..., N, D]",
    "return: [broadcast batch..., broadcast N, broadcast P]",
)
def evaluate_constant_or_function(
    value: ParameterOrFunction,
    X: TensorType,
    *,
    lower_bound: Optional[float] = None,
) -> TensorType:
    if isinstance(value, Function):
        result = value(X)
        if lower_bound is not None:
            result = tf.maximum(result, lower_bound)
        return result
    else:
        return value


class Linear(MeanFunction, Function):
    """
    y_i = A x_i + b
    """

    def __init__(self, A: TensorType = None, b: TensorType = None) -> None:
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be [D, Q], b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

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
    def __init__(self, c: TensorType = None) -> None:
        super().__init__()
        c = np.zeros(1) if c is None else c
        self.c = Parameter(c)

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

    def __call__(self, X: TensorType) -> tf.Tensor:
        output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=X.dtype)


class Polynomial(MeanFunction, Function):
    @check_shapes("w: [broadcast n_terms, broadcast output_dim]")
    def __init__(
        self, degree: int, input_dim: int = 1, output_dim: int = 1, w: Optional[TensorType] = None
    ) -> None:
        powers = cs(tuple(self._compute_powers(degree, input_dim)), "[n_terms, input_dim]")
        if w is None:
            w = cs([[1.0]] + (len(powers) - 1) * [[0.0]], "[n_terms, 1]")
        w_shape = (len(powers), output_dim)
        self.powers = tf.constant(powers, dtype=default_float())
        self.w = Parameter(tf.broadcast_to(w, w_shape))

    def _compute_powers(self, degree: int, input_dim: int) -> Iterator[Tuple[int, ...]]:
        if not input_dim:
            yield ()
            return
        for i in range(degree + 1):
            for inner in self._compute_powers(degree - i, input_dim - 1):
                yield (i,) + inner

    @check_shapes(
        "X: [batch..., input_dim]",
        "return: [batch..., output_dim]",
    )
    def __call__(self, X: TensorType) -> tf.Tensor:
        raised = cs(tf.pow(X[..., None, :], self.powers), "[batch..., n_terms, input_dim]")
        prod = cs(tf.math.reduce_prod(raised, axis=-1), "[batch..., n_terms]")
        return tf.einsum("...i,ij->...j", prod, self.w)


class SwitchedMeanFunction(MeanFunction, Function):
    """
    This class enables to use different (independent) mean_functions respective
    to the data 'label'.
    We assume the 'label' is stored in the extra column of X.
    """

    def __init__(self, meanfunction_list: Collection[MeanFunction]) -> None:
        super().__init__()
        for m in meanfunction_list:
            assert isinstance(m, MeanFunction)
        self.meanfunctions = meanfunction_list

    def __call__(self, X: TensorType) -> tf.Tensor:
        ind = tf.gather(tf.transpose(X), tf.shape(X)[1] - 1)  # ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(
            tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1] - 1))
        )  # X = X[:,:-1]

        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, len(self.meanfunctions))
        # apply the likelihood-function to each section of the data
        results = [m(x) for x, m in zip(x_list, self.meanfunctions)]
        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.meanfunctions))
        return tf.dynamic_stitch(partitions, results)
