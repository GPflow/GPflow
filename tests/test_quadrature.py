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

import gpflow
from gpflow.test_util import session_tf


@pytest.fixture
def mu1(): return np.array([1.0, 1.3])

@pytest.fixture
def mu2(): return np.array([-2.0, 0.3])

@pytest.fixture
def var1(): return np.array([3.0, 3.5])

@pytest.fixture
def var2(): return np.array([4.0, 4.2])

def cast(x):
    return tf.cast(np.asarray(x), dtype=gpflow.settings.float_type)


def test_diagquad_1d(session_tf, mu1, var1):
    quad = gpflow.quadrature.ndiagquad(
            [lambda *X: tf.exp(X[0])], 25,
            [cast(mu1)], [cast(var1)])
    res, = session_tf.run(quad)
    expected = np.exp(mu1 + var1/2)
    assert_allclose(res, expected)


def test_diagquad_2d(session_tf, mu1, var1, mu2, var2):
    alpha = 2.5
    quad = gpflow.quadrature.ndiagquad(
            lambda *X: tf.exp(X[0] + alpha * X[1]),
            35,  # using logspace=True we can reduce this, see test_diagquad_logspace
            [cast(mu1), cast(mu2)], [cast(var1), cast(var2)])
    res = session_tf.run(quad)
    expected = np.exp(mu1 + var1/2 + alpha * mu2 + alpha**2 * var2/2)
    assert_allclose(res, expected)


def test_diagquad_logspace(session_tf, mu1, var1, mu2, var2):
    alpha = 2.5
    quad = gpflow.quadrature.ndiagquad(
            lambda *X: (X[0] + alpha * X[1]),
            25,
            [cast(mu1), cast(mu2)], [cast(var1), cast(var2)],
            logspace=True)
    res = session_tf.run(quad)
    expected = mu1 + var1/2 + alpha * mu2 + alpha**2 * var2/2
    assert_allclose(res, expected)


def test_diagquad_with_kwarg(session_tf, mu2, var2):
    alpha = np.array([2.5, -1.3])
    quad = gpflow.quadrature.ndiagquad(
            lambda X, Y: tf.exp(X * Y), 25,
            cast(mu2), cast(var2), Y=alpha)
    res = session_tf.run(quad)
    expected = np.exp(alpha * mu2 + alpha**2 * var2/2)
    assert_allclose(res, expected)
