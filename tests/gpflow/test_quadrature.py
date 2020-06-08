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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow.quadrature as quadrature


@pytest.mark.parametrize("mu", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var", [np.array([3.0, 3.5])])
def test_diagquad_1d(mu, var):
    num_gauss_hermite_points = 25

    mu = mu[:, None]
    var = var[:, None]

    # quad = quadrature.ndiagquad([lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var])
    quad = quadrature.ndiagquad([lambda *X: tf.exp(X[0])], num_gauss_hermite_points, mu, var)

    expected = np.exp(mu + var / 2)
    assert_allclose(quad[0], expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_2d(mu1, var1, mu2, var2):
    alpha = 2.5
    # using logspace=True we can reduce this, see test_diagquad_logspace
    num_gauss_hermite_points = 35

    mu = np.hstack([x[:, None] for x in (mu1, mu2)])
    var = np.hstack([x[:, None] for x in (var1, var2)])

    # fun = lambda *X: tf.exp(X[0] + alpha * X[1])
    fun = lambda X: tf.exp(X[..., 0:1] + alpha * X[..., 1:2])

    # quad = quadrature.ndiagquad(fun, num_gauss_hermite_points, [mu1, mu2], [var1, var2],)
    quad = quadrature.ndiagquad(fun, num_gauss_hermite_points, mu, var,)

    # expected = np.exp(mu1 + var1 / 2 + alpha * mu2 + alpha ** 2 * var2 / 2)
    expected = np.exp(
        mu[..., 0:1] + var[..., 0:1] / 2 + alpha * mu[..., 1:2] + alpha ** 2 * var[..., 1:2] / 2
    )

    assert_allclose(quad, expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_logspace(mu1, var1, mu2, var2):
    alpha = 2.5
    num_gauss_hermite_points = 25

    mu = np.hstack([x[:, None] for x in (mu1, mu2)])
    var = np.hstack([x[:, None] for x in (var1, var2)])

    # fun = lambda *X: (X[0] + alpha * X[1])
    fun = lambda X: (X[..., 0:1] + alpha * X[..., 1:2])

    # quad = quadrature.ndiagquad(
    #     fun, num_gauss_hermite_points, [mu1, mu2], [var1, var2], logspace=True,
    # )
    quad = quadrature.ndiagquad(fun, num_gauss_hermite_points, mu, var, logspace=True,)

    # expected = mu1 + var1 / 2 + alpha * mu2 + alpha ** 2 * var2 / 2
    expected = (
        mu[..., 0:1] + var[..., 0:1] / 2 + alpha * mu[..., 1:2] + alpha ** 2 * var[..., 1:2] / 2
    )

    assert_allclose(quad, expected)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
def test_diagquad_with_kwarg(mu1, var1):
    alpha = np.array([2.5, -1.3])
    num_gauss_hermite_points = 25

    mu1 = mu1[:, None]
    var1 = var1[:, None]
    alpha = alpha[:, None]

    fun = lambda X, Y: tf.exp(X * Y)
    quad = quadrature.ndiagquad(fun, num_gauss_hermite_points, mu1, var1, Y=alpha)

    expected = np.exp(alpha * mu1 + alpha ** 2 * var1 / 2)

    print(expected)

    assert_allclose(quad, expected)
