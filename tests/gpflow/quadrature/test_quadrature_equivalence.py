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
from ndiagquad_old import ndiagquad as ndiagquad_old
from numpy.testing import assert_allclose

from gpflow.quadrature import ndiagquad


@pytest.mark.parametrize("mu", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var", [np.array([3.0, 3.5])])
def test_diagquad_1d(mu, var):
    num_gauss_hermite_points = 25
    quad = ndiagquad([lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var])
    quad_old = ndiagquad_old([lambda *X: tf.exp(X[0])], num_gauss_hermite_points, [mu], [var])
    assert_allclose(quad[0], quad_old[0])


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_2d(mu1, var1, mu2, var2):
    alpha = 2.5
    # using logspace=True we can reduce this, see test_diagquad_logspace
    num_gauss_hermite_points = 35
    quad = ndiagquad(
        lambda *X: tf.exp(X[0] + alpha * X[1]), num_gauss_hermite_points, [mu1, mu2], [var1, var2],
    )
    quad_old = ndiagquad_old(
        lambda *X: tf.exp(X[0] + alpha * X[1]), num_gauss_hermite_points, [mu1, mu2], [var1, var2],
    )
    assert_allclose(quad, quad_old)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
@pytest.mark.parametrize("mu2", [np.array([-2.0, 0.3])])
@pytest.mark.parametrize("var2", [np.array([4.0, 4.2])])
def test_diagquad_logspace(mu1, var1, mu2, var2):
    alpha = 2.5
    num_gauss_hermite_points = 25
    quad = ndiagquad(
        lambda *X: (X[0] + alpha * X[1]),
        num_gauss_hermite_points,
        [mu1, mu2],
        [var1, var2],
        logspace=True,
    )
    quad_old = ndiagquad_old(
        lambda *X: (X[0] + alpha * X[1]),
        num_gauss_hermite_points,
        [mu1, mu2],
        [var1, var2],
        logspace=True,
    )
    assert_allclose(quad, quad_old)


@pytest.mark.parametrize("mu1", [np.array([1.0, 1.3])])
@pytest.mark.parametrize("var1", [np.array([3.0, 3.5])])
def test_diagquad_with_kwarg(mu1, var1):
    alpha = np.array([2.5, -1.3])
    num_gauss_hermite_points = 25
    quad = ndiagquad(lambda X, Y: tf.exp(X * Y), num_gauss_hermite_points, mu1, var1, Y=alpha)
    quad_old = ndiagquad_old(
        lambda X, Y: tf.exp(X * Y), num_gauss_hermite_points, mu1, var1, Y=alpha
    )
    assert_allclose(quad, quad_old)
