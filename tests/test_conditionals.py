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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.conditionals import conditional
from gpflow.utilities.defaults import default_float

rng = np.random.RandomState(123)

Ln = 2
Nn = 10
Mn = 50


@pytest.fixture(scope='module')
def kernel():
    k = gpflow.kernels.Matern32() + gpflow.kernels.White()
    k.kernels[1].variance <<= 0.01
    return k


@pytest.fixture(scope='module')
def Xdata():
    return tf.convert_to_tensor(rng.randn(Nn, 1))


@pytest.fixture(scope='module')
def Xnew():
    return tf.convert_to_tensor(rng.randn(Mn, 1))


@pytest.fixture(scope='module')
def mu():
    return tf.convert_to_tensor(rng.randn(Nn, Ln))


@pytest.fixture(scope='module')
def sqrt():
    return tf.convert_to_tensor(rng.randn(Nn, Ln))


@pytest.fixture(scope='module')
def chol(sqrt):
    return tf.stack([tf.linalg.diag(sqrt[:, i]) for i in range(Ln)])


@pytest.mark.parametrize('white', [True, False])
def test_diag(Xdata, Xnew, kernel, mu, sqrt, chol, white):
    Fstar_mean_1, Fstar_var_1 = conditional(Xnew,
                                            Xdata,
                                            kernel,
                                            mu,
                                            q_sqrt=sqrt,
                                            white=white)
    Fstar_mean_2, Fstar_var_2 = conditional(Xnew,
                                            Xdata,
                                            kernel,
                                            mu,
                                            q_sqrt=chol,
                                            white=white)

    mean_diff = Fstar_mean_1 - Fstar_mean_2
    var_diff = Fstar_var_1 - Fstar_var_2

    assert_allclose(mean_diff, 0)
    assert_allclose(var_diff, 0)


def test_whiten(Xdata, Xnew, kernel, mu, sqrt):
    """
    Make sure that predicting using the whitened representation is the
    sameas the non-whitened one.
    """

    K = kernel(Xdata) + tf.eye(Nn, dtype=default_float()) * 1e-6
    L = tf.linalg.cholesky(K)
    V = tf.linalg.triangular_solve(L, mu, lower=True)
    mean1, var1 = conditional(Xnew, Xdata, kernel, mu)
    mean2, var2 = conditional(Xnew, Xdata, kernel, V, white=True)

    assert_allclose(mean1, mean2)
    assert_allclose(var1, var2)


def test_gaussian_whiten(Xdata, Xnew, kernel, mu, sqrt):
    """
    Make sure that predicting using the whitened representation is the
    same as the non-whitened one.
    """
    F_sqrt = tf.convert_to_tensor(rng.rand(Nn, Ln))

    K = kernel(Xdata)
    L = tf.linalg.cholesky(K)
    V = tf.linalg.triangular_solve(L, mu, lower=True)
    V_prime = tf.linalg.diag(tf.transpose(F_sqrt))
    common_shape = tf.broadcast_static_shape(V_prime.shape, L.shape)
    L = tf.broadcast_to(L, common_shape)
    V_sqrt = tf.linalg.triangular_solve(L,
                                        tf.linalg.diag(tf.transpose(F_sqrt)),
                                        lower=True)

    Fstar_mean, Fstar_var = conditional(Xnew, Xdata, kernel, mu, q_sqrt=F_sqrt)
    Fstar_w_mean, Fstar_w_var = conditional(Xnew,
                                            Xdata,
                                            kernel,
                                            V,
                                            q_sqrt=V_sqrt,
                                            white=True)

    mean_diff = Fstar_w_mean - Fstar_mean
    var_diff = Fstar_w_var - Fstar_var

    assert_allclose(mean_diff, 0, atol=4)
    assert_allclose(var_diff, 0, atol=4)
