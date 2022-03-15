# Copyright 2018 the GPflow authors.
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

from typing import cast

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow.quadrature as quadrature
from gpflow.base import AnyNDArray, TensorType


@pytest.mark.parametrize("mu", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var", [np.array([3.0, 3.5])])
def test_diagquad_1d(mu: TensorType, var: TensorType) -> None:
    num_gauss_hermite_points = 25
    quad = quadrature.ndiagquad([lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var])
    expected = np.exp(mu + var / 2)
    assert_allclose(quad[0], expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_2d(mu1: TensorType, var1: TensorType, mu2: TensorType, var2: TensorType) -> None:
    alpha = 2.5
    # using logspace=True we can reduce this, see test_diagquad_logspace
    num_gauss_hermite_points = 35
    quad = quadrature.ndiagquad(
        lambda *X: tf.exp(X[0] + alpha * X[1]),
        num_gauss_hermite_points,
        [mu1, mu2],
        [var1, var2],
    )
    expected = np.exp(mu1 + var1 / 2 + alpha * mu2 + alpha ** 2 * var2 / 2)
    assert_allclose(quad, expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_logspace(
    mu1: TensorType, var1: TensorType, mu2: TensorType, var2: TensorType
) -> None:
    alpha = 2.5
    num_gauss_hermite_points = 25
    quad = quadrature.ndiagquad(
        lambda *X: (X[0] + alpha * X[1]),
        num_gauss_hermite_points,
        [mu1, mu2],
        [var1, var2],
        logspace=True,
    )
    expected = mu1 + var1 / 2 + alpha * mu2 + alpha ** 2 * var2 / 2
    assert_allclose(quad, expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
def test_diagquad_with_kwarg(mu1: TensorType, var1: TensorType) -> None:
    alpha: AnyNDArray = np.array([2.5, -1.3])
    num_gauss_hermite_points = 25
    quad = quadrature.ndiagquad(
        lambda X, Y: tf.exp(X * Y), num_gauss_hermite_points, mu1, var1, Y=alpha
    )
    expected = np.exp(alpha * mu1 + alpha ** 2 * var1 / 2)
    assert_allclose(quad, expected)


def test_ndiagquad_does_not_throw_error() -> None:
    """
    Check that the autograph=False for quadrature.ndiagquad does not throw an error.
    Regression test for https://github.com/GPflow/GPflow/issues/1547.
    """

    @tf.function(autograph=False)
    def func_ndiagquad_autograph_false() -> tf.Tensor:
        mu: AnyNDArray = np.array([1.0, 1.3])
        var: AnyNDArray = np.array([3.0, 3.5])
        num_gauss_hermite_points = 25
        return quadrature.ndiagquad(
            [lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var]
        )

    func_ndiagquad_autograph_false()


def test_quadrature_autograph() -> None:
    """
    Check that the return value is equal with and without Autograph
    Regression test for https://github.com/GPflow/GPflow/issues/1547.
    """

    def compute(autograph: bool) -> AnyNDArray:
        @tf.function(autograph=autograph)
        def func() -> tf.Tensor:
            mu: AnyNDArray = np.array([1.0, 1.3])
            var: AnyNDArray = np.array([3.0, 3.5])
            num_gauss_hermite_points = 25
            return quadrature.ndiagquad(
                [lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var]
            )

        (result,) = func()
        return cast(AnyNDArray, result.numpy())

    np.testing.assert_equal(
        compute(autograph=True),
        compute(autograph=False),
    )
