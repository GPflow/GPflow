# Copyright 2017 the GPflow authors.
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

from typing import Any, Sequence, Tuple, Type

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gpflow
from gpflow.base import AnyNDArray, TensorType
from gpflow.config import default_int
from gpflow.experimental.check_shapes import check_shapes
from gpflow.functions import (
    Additive,
    Constant,
    Linear,
    MeanFunction,
    Polynomial,
    Product,
    SwitchedMeanFunction,
    Zero,
)
from gpflow.inducing_variables import InducingPoints

rng = np.random.RandomState(99021)


class Datum:
    input_dim, output_dim = 3, 2
    N, Ntest, M = 20, 30, 10


_mean_functions = [
    Zero(),
    Linear(
        A=rng.randn(Datum.input_dim, Datum.output_dim),
        b=rng.randn(Datum.output_dim, 1).reshape(-1),
    ),
    Constant(c=rng.randn(Datum.output_dim, 1).reshape(-1)),
    Polynomial(degree=2, input_dim=Datum.input_dim, output_dim=Datum.output_dim),
]


@pytest.mark.parametrize("mean_function_1", _mean_functions)
@pytest.mark.parametrize("mean_function_2", _mean_functions)
@pytest.mark.parametrize("operation", ["+", "*"])
def test_mean_functions_output_shape(
    mean_function_1: MeanFunction, mean_function_2: MeanFunction, operation: str
) -> None:
    """
    Test the output shape for basic and compositional mean functions, also
    check that the combination of mean functions returns the correct class
    """
    X = np.random.randn(Datum.N, Datum.input_dim)
    Y = mean_function_1(X)
    # basic output shape check
    assert Y.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]

    # composed mean function output shape check
    if operation == "+":
        mean_composed = mean_function_1 + mean_function_2
    elif operation == "*":
        mean_composed = mean_function_1 * mean_function_2
    else:
        raise NotImplementedError

    Y_composed = mean_composed(X)
    assert Y_composed.shape in [(Datum.N, Datum.output_dim), (Datum.N, 1)]


@pytest.mark.parametrize("mean_function_1", _mean_functions)
@pytest.mark.parametrize("mean_function_2", _mean_functions)
@pytest.mark.parametrize("operation", ["+", "*"])
def test_mean_functions_composite_type(
    mean_function_1: MeanFunction, mean_function_2: MeanFunction, operation: str
) -> None:
    if operation == "+":
        mean_composed = mean_function_1 + mean_function_2
        assert isinstance(mean_composed, Additive)
    elif operation == "*":
        mean_composed = mean_function_1 * mean_function_2
        assert isinstance(mean_composed, Product)
    else:
        raise NotImplementedError


_linear_functions = [
    Linear(
        A=rng.randn(Datum.input_dim, Datum.output_dim),
        b=rng.randn(Datum.output_dim, 1).reshape(-1),
    )
    for _ in range(3)
]

# Append inverse of first Linear mean function in _linear_functions
_linear_functions.append(Linear(A=-1.0 * _linear_functions[0].A, b=-1.0 * _linear_functions[0].b))

_constant_functions = [Constant(c=rng.randn(Datum.output_dim, 1).reshape(-1)) for _ in range(3)]
# Append inverse of first Constant mean function in _constant_functions
_constant_functions.append(Constant(c=-1.0 * _constant_functions[0].c))


@check_shapes(
    "X: [N, D]",
    "Y: [N, Q]",
)
def _create_GPR_model_with_bias(
    X: TensorType, Y: TensorType, mean_function: MeanFunction
) -> gpflow.models.GPR:
    return gpflow.models.GPR(
        (X, Y), mean_function=mean_function, kernel=gpflow.kernels.Bias(Datum.input_dim)
    )


@pytest.mark.parametrize("mean_functions", [_linear_functions, _constant_functions])
def test_mean_functions_distributive_property(mean_functions: Sequence[MeanFunction]) -> None:
    """
    Tests that distributive property of addition and multiplication holds for mean functions
    (both Constant and Linear): A * (B + C) = A * B + A * C
    """
    X, Y = rng.randn(Datum.N, Datum.input_dim), rng.randn(Datum.N, Datum.output_dim)
    Xtest = rng.randn(30, Datum.input_dim)
    A, B, C = mean_functions[0], mean_functions[1], mean_functions[2]
    lhs = Product(A, Additive(B, C))  # A * (B + C)
    rhs = Additive(Product(A, B), Product(A, C))  # A * B + A * C

    model_lhs = _create_GPR_model_with_bias(X, Y, mean_function=lhs)
    model_rhs = _create_GPR_model_with_bias(X, Y, mean_function=rhs)

    mu_lhs, var_lhs = model_lhs.predict_f(Xtest)
    mu_rhs, var_rhs = model_rhs.predict_f(Xtest)

    assert_allclose(mu_lhs, mu_rhs)
    assert_allclose(var_lhs, var_rhs)


@pytest.mark.parametrize("mean_functions", [_linear_functions, _constant_functions])
def test_mean_functions_A_minus_A_equals_zero(mean_functions: Sequence[MeanFunction]) -> None:
    """
    Tests that the addition the inverse of a mean function to itself is equivalent to having a
    Zero mean function: A + (-A) = 0
    """
    X, Y = rng.randn(Datum.N, Datum.input_dim), rng.randn(Datum.N, Datum.output_dim)
    Xtest = rng.randn(30, Datum.input_dim)
    A, A_inverse = mean_functions[0], mean_functions[-1]
    lhs = Additive(A, A_inverse)  # A + (-A)
    rhs = Zero()  # 0

    model_lhs = _create_GPR_model_with_bias(X, Y, mean_function=lhs)
    model_rhs = _create_GPR_model_with_bias(X, Y, mean_function=rhs)

    mu_lhs, var_lhs = model_lhs.predict_f(Xtest)
    mu_rhs, var_rhs = model_rhs.predict_f(Xtest)

    assert_allclose(mu_lhs, mu_rhs)
    assert_allclose(var_lhs, var_rhs)


@pytest.mark.parametrize("mean_functions", [_linear_functions])
def test_linear_mean_functions_associative_property(mean_functions: Sequence[MeanFunction]) -> None:
    """
    Tests that associative property of addition holds for linear mean functions:
    A + (B + (-A)) = B = (A + B) + (-A)
    """
    X, Y = rng.randn(Datum.N, Datum.input_dim), rng.randn(Datum.N, Datum.output_dim)
    Xtest = rng.randn(30, Datum.input_dim)
    A, B, A_inverse = mean_functions[0], mean_functions[1], mean_functions[-1]

    lhs = Additive(A, Additive(B, A_inverse))  # A + (B + (-A))
    rhs = Additive(Additive(A, B), A_inverse)  # (A + B) + (-A)

    model_lhs = _create_GPR_model_with_bias(X, Y, mean_function=lhs)
    model_b = _create_GPR_model_with_bias(X, Y, mean_function=B)
    model_rhs = _create_GPR_model_with_bias(X, Y, mean_function=rhs)

    mu_lhs, var_lhs = model_lhs.predict_f(Xtest)
    mu_b, var_b = model_b.predict_f(Xtest)
    mu_rhs, var_rhs = model_rhs.predict_f(Xtest)

    assert_allclose(mu_lhs, mu_b)
    assert_allclose(var_lhs, var_b)
    assert_allclose(mu_b, mu_rhs)
    assert_allclose(var_b, var_rhs)


@pytest.mark.parametrize("batch", [(), (1, 2)])
@pytest.mark.parametrize("degree", [0, 1, 3])
@pytest.mark.parametrize("input_dim", [0, 1, 3])
@pytest.mark.parametrize("output_dim", [1, 2])
def test_polynomial__sanity(
    batch: Tuple[int, ...], degree: int, input_dim: int, output_dim: int
) -> None:
    p = Polynomial(degree, input_dim, output_dim)
    X = np.ones(batch + (input_dim,))
    Y = p(X)
    assert (batch) + (output_dim,) == Y.shape
    assert_allclose(1.0, Y)  # Polynomial is initialised to constant 1.0.


def test_polynomial__compute_powers() -> None:
    assert_allclose(
        [
            (0, 0, 0),
            (0, 0, 1),
            (0, 0, 2),
            (0, 1, 0),
            (0, 1, 1),
            (0, 2, 0),
            (1, 0, 0),
            (1, 0, 1),
            (1, 1, 0),
            (2, 0, 0),
        ],
        list(Polynomial.compute_powers(degree=2, input_dim=3)),
    )


def test_polynomial__1d() -> None:
    # Test on a 1D quadratic function, where we can easily work out the maths.
    p = Polynomial(degree=2, w=[1.0, 2.0, 3.0])
    X = np.array([[1.0], [2.0]])
    Y = p(X)
    assert_allclose(
        [
            [1.0 + 2.0 * 1.0 + 3.0 * (1.0 ** 2)],
            [1.0 + 2.0 * 2.0 + 3.0 * (2.0 ** 2)],
        ],
        Y,
    )


def test_polynomial__linear() -> None:
    # Test on a 3D linear function, where we can easily work out the maths.
    p = Polynomial(degree=1, input_dim=3, w=[1.0, 2.0, 3.0, 4.0])
    X = np.array([1.0, 2.0, 3.0])
    Y = p(X)
    assert_allclose([1.0 + 2.0 * 3.0 + 3.0 * 2.0 + 4.0 * 1.0], Y)


@pytest.mark.parametrize("N, D", [[10, 3]])
def test_switched_mean_function(N: int, D: int) -> None:
    """
    Test for the SwitchedMeanFunction.
    """
    X: AnyNDArray = np.hstack([rng.randn(N, D), 1.0 * rng.randint(0, 2, N).reshape(-1, 1)])
    zeros, ones = Constant(np.zeros(1)), Constant(np.ones(1))
    switched_mean = SwitchedMeanFunction([zeros, ones])

    np_list: AnyNDArray = np.array([0.0, 1.0])
    result_ref = (np_list[X[:, D].astype(default_int())]).reshape(-1, 1)
    result = switched_mean(X)

    assert_allclose(result, result_ref)


def test_bug_277_regression() -> None:
    """
    See github issue #277. This is a regression test.
    """
    model1, model2 = Linear(), Linear()
    assert model1.b.numpy() == model2.b.numpy()
    model2.b.assign([1.0])
    assert not model1.b.numpy() == model2.b.numpy()


_model_classes = [
    gpflow.models.GPR,
    gpflow.models.SGPR,
    gpflow.models.GPRFITC,
    gpflow.models.SVGP,
    gpflow.models.VGP,
    gpflow.models.GPMC,
    gpflow.models.SGPMC,
]


@pytest.mark.parametrize("model_class", _model_classes)
def test_models_with_mean_functions_changes(model_class: Type[Any]) -> None:
    """
    Simply check that all models have a higher prediction with a constant mean
    function than with a zero mean function.

    For compositions of mean functions check that multiplication/ addition of
    a constant results in a higher prediction, whereas addition of zero/
    mutliplication with one does not.
    """
    data = rng.randn(Datum.N, Datum.input_dim), rng.randn(Datum.N, 1)
    Xnew = rng.randn(Datum.Ntest, Datum.input_dim)
    inducing_variable = InducingPoints(Z=rng.randn(Datum.M, Datum.input_dim))
    kernel = gpflow.kernels.Matern32()
    likelihood = gpflow.likelihoods.Gaussian()
    zero_mean = Zero()
    non_zero_mean = Constant(c=np.ones(1) * 10)

    if model_class in [gpflow.models.GPR]:
        model_zero_mean = model_class(data, kernel=kernel, mean_function=zero_mean)
        model_non_zero_mean = model_class(data, kernel=kernel, mean_function=non_zero_mean)
    elif model_class in [gpflow.models.VGP]:
        model_zero_mean = model_class(
            data, likelihood=likelihood, kernel=kernel, mean_function=zero_mean
        )
        model_non_zero_mean = model_class(
            data, likelihood=likelihood, kernel=kernel, mean_function=non_zero_mean
        )
    elif model_class in [gpflow.models.SVGP]:
        model_zero_mean = model_class(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=zero_mean,
        )
        model_non_zero_mean = model_class(
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=non_zero_mean,
        )
    elif model_class in [gpflow.models.SGPR, gpflow.models.GPRFITC]:
        model_zero_mean = model_class(
            data,
            kernel=kernel,
            inducing_variable=inducing_variable,
            mean_function=zero_mean,
        )
        model_non_zero_mean = model_class(
            data,
            kernel=kernel,
            inducing_variable=inducing_variable,
            mean_function=non_zero_mean,
        )
    elif model_class in [gpflow.models.SGPMC]:
        model_zero_mean = model_class(
            data,
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=zero_mean,
        )
        model_non_zero_mean = model_class(
            data,
            kernel=kernel,
            likelihood=likelihood,
            inducing_variable=inducing_variable,
            mean_function=non_zero_mean,
        )
    elif model_class in [gpflow.models.GPMC]:
        model_zero_mean = model_class(
            data, kernel=kernel, likelihood=likelihood, mean_function=zero_mean
        )
        model_non_zero_mean = model_class(
            data, kernel=kernel, likelihood=likelihood, mean_function=non_zero_mean
        )
    else:
        raise NotImplementedError

    mu_zero, var_zero = model_zero_mean.predict_f(Xnew)
    mu_non_zero, var_non_zero = model_non_zero_mean.predict_f(Xnew)
    # predictive variance remains unchanged after modifying mean function
    assert np.all(var_zero.numpy() == var_non_zero.numpy())
    # predictive mean changes after modifying mean function
    assert not np.all(mu_zero.numpy() == mu_non_zero.numpy())
