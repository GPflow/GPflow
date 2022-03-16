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
from gpflow import Parameter
from gpflow.base import AnyNDArray, MeanAndVariance
from gpflow.conditionals import conditional
from gpflow.config import default_float
from gpflow.kernels import Kernel
from gpflow.utilities.bijectors import triangular

rng = np.random.RandomState(123)

Ln = 2
Nn = 10
Mn = 20


@pytest.fixture(scope="module")
def kernel() -> Kernel:
    k = gpflow.kernels.Matern32() + gpflow.kernels.White()
    k.kernels[1].variance.assign(0.01)
    return k


@pytest.fixture(scope="module")
def Xdata() -> tf.Tensor:
    return tf.convert_to_tensor(rng.randn(Nn, 1))


@pytest.fixture(scope="module")
def Xnew() -> tf.Tensor:
    return tf.convert_to_tensor(rng.randn(Mn, 1))


@pytest.fixture(scope="module")
def mu() -> tf.Tensor:
    return tf.convert_to_tensor(rng.randn(Nn, Ln))


@pytest.fixture(scope="module")
def sqrt() -> tf.Tensor:
    return tf.convert_to_tensor(rng.randn(Nn, Ln))


@pytest.fixture(scope="module")
def chol(sqrt: tf.Tensor) -> tf.Tensor:
    return tf.stack([tf.linalg.diag(sqrt[:, i]) for i in range(Ln)])


@pytest.mark.parametrize("white", [True, False])
def test_diag(
    Xdata: tf.Tensor,
    Xnew: tf.Tensor,
    kernel: Kernel,
    mu: tf.Tensor,
    sqrt: tf.Tensor,
    chol: tf.Tensor,
    white: bool,
) -> None:
    Fstar_mean_1, Fstar_var_1 = conditional(Xnew, Xdata, kernel, mu, q_sqrt=sqrt, white=white)
    Fstar_mean_2, Fstar_var_2 = conditional(Xnew, Xdata, kernel, mu, q_sqrt=chol, white=white)

    mean_diff = Fstar_mean_1 - Fstar_mean_2
    var_diff = Fstar_var_1 - Fstar_var_2

    assert_allclose(mean_diff, 0)
    assert_allclose(var_diff, 0)


def test_whiten(
    Xdata: tf.Tensor, Xnew: tf.Tensor, kernel: Kernel, mu: tf.Tensor, sqrt: tf.Tensor
) -> None:
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


def test_gaussian_whiten(
    Xdata: tf.Tensor, Xnew: tf.Tensor, kernel: Kernel, mu: tf.Tensor, sqrt: tf.Tensor
) -> None:
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
    V_sqrt = tf.linalg.triangular_solve(L, tf.linalg.diag(tf.transpose(F_sqrt)), lower=True)

    Fstar_mean, Fstar_var = conditional(Xnew, Xdata, kernel, mu, q_sqrt=F_sqrt)
    Fstar_w_mean, Fstar_w_var = conditional(Xnew, Xdata, kernel, V, q_sqrt=V_sqrt, white=True)

    mean_diff = Fstar_w_mean - Fstar_mean
    var_diff = Fstar_w_var - Fstar_var

    assert_allclose(mean_diff, 0, atol=4)
    assert_allclose(var_diff, 0, atol=4)


@pytest.mark.parametrize("white", [True, False])
def test_q_sqrt_constraints(
    Xdata: tf.Tensor, Xnew: tf.Tensor, kernel: Kernel, mu: tf.Tensor, white: bool
) -> None:
    """Test that sending in an unconstrained q_sqrt returns the same conditional
    evaluation and gradients. This is important to match the behaviour of the KL, which
    enforces q_sqrt is triangular.
    """

    tril: AnyNDArray = np.tril(rng.randn(Ln, Nn, Nn))

    q_sqrt_constrained = Parameter(tril, transform=triangular())
    q_sqrt_unconstrained = Parameter(tril)

    diff_before_gradient_step = (q_sqrt_constrained - q_sqrt_unconstrained).numpy()
    assert_allclose(diff_before_gradient_step, 0)

    Fstars = []
    for q_sqrt in [q_sqrt_constrained, q_sqrt_unconstrained]:

        with tf.GradientTape() as tape:
            _, Fstar_var = conditional(Xnew, Xdata, kernel, mu, q_sqrt=q_sqrt, white=white)

        grad = tape.gradient(Fstar_var, q_sqrt.unconstrained_variable)
        q_sqrt.unconstrained_variable.assign_sub(grad)
        Fstars.append(Fstar_var)

    diff_Fstar_before_gradient_step = Fstars[0] - Fstars[1]
    assert_allclose(diff_Fstar_before_gradient_step, 0)

    diff_after_gradient_step = (q_sqrt_constrained - q_sqrt_unconstrained).numpy()
    assert_allclose(diff_after_gradient_step, 0)


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("features_inducing_points", [False, True])
def test_base_conditional_vs_ref(full_cov: bool, features_inducing_points: bool) -> None:
    """
    Test that conditionals agree with a slow-but-clear numpy implementation
    """
    Dy, N, M, Dx = 5, 4, 3, 2
    X = np.random.randn(N, Dx)
    Z = np.random.randn(M, Dx)
    kern = gpflow.kernels.Matern52(lengthscales=0.5)
    q_mu = np.random.randn(M, Dy)
    q_sqrt: AnyNDArray = np.tril(np.random.randn(Dy, M, M), -1)

    def numpy_conditional(
        X: tf.Tensor, Z: tf.Tensor, kern: Kernel, q_mu: tf.Tensor, q_sqrt: tf.Tensor
    ) -> MeanAndVariance:
        Kmm = kern(Z, Z) + np.eye(M) * gpflow.config.default_jitter()
        Kmn = kern(Z, X)
        Knn = kern(X, X)

        Kmm, Kmn, Knn = [k.numpy() for k in [Kmm, Kmn, Knn]]
        Knm: AnyNDArray = Kmn.T

        Kmm, Kmn, Knm, Knn = [np.tile(k[None, :, :], [Dy, 1, 1]) for k in [Kmm, Kmn, Knm, Knn]]

        S = q_sqrt @ np.transpose(q_sqrt, [0, 2, 1])

        Kmm_inv = np.linalg.inv(Kmm)
        mean = np.einsum("dmn,dmM,Md->nd", Kmn, Kmm_inv, q_mu)
        cov = Knn + Knm @ Kmm_inv @ (S - Kmm) @ Kmm_inv @ Kmn
        return mean, cov

    mean_np, cov_np = numpy_conditional(X, Z, kern, q_mu, q_sqrt)

    if features_inducing_points:
        Z = gpflow.inducing_variables.InducingPoints(Z)

    mean_gpflow, cov_gpflow = [
        v.numpy()
        for v in gpflow.conditionals.conditional(
            X, Z, kern, q_mu, q_sqrt=tf.identity(q_sqrt), white=False, full_cov=full_cov
        )
    ]

    if not full_cov:
        cov_np = np.diagonal(cov_np, axis1=-1, axis2=-2).T

    assert_allclose(mean_np, mean_gpflow)
    assert_allclose(cov_np, cov_gpflow)
