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

# -*- coding: utf-8 -*-

from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose, assert_almost_equal

import gpflow
from gpflow import Parameter, default_float, default_jitter
from gpflow.base import AnyNDArray, TensorType
from gpflow.experimental.check_shapes import ShapeChecker, check_shapes
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel
from gpflow.kullback_leiblers import gauss_kl, prior_kl
from gpflow.utilities.bijectors import triangular

rng = np.random.RandomState(0)

# ------------------------------------------
# Fixtures
# ------------------------------------------

Ln = 2
Nn = 10
Mn = 50


@pytest.fixture(scope="module")
def kernel() -> Kernel:
    k = gpflow.kernels.Matern32() + gpflow.kernels.White()
    k.kernels[1].variance.assign(0.01)
    return k


@pytest.fixture(scope="module")
@check_shapes(
    "return: [N, 1, 1]",
)
def inducing_points() -> InducingPoints:
    return InducingPoints(rng.randn(Nn, 1))


@pytest.fixture(scope="module")
@check_shapes(
    "return: [N, L]",
)
def mu() -> Parameter:
    return Parameter(rng.randn(Nn, Ln))


# ------------------------------------------
# Helpers
# ------------------------------------------


@check_shapes(
    "return: [N, M, M]",
)
def make_sqrt(N: int, M: int) -> TensorType:
    return np.array([np.tril(rng.randn(M, M)) for _ in range(N)])


@check_shapes(
    "return: [N, M, M]",
)
def make_K_batch(N: int, M: int) -> TensorType:
    K_np = rng.randn(N, M, M)
    beye: AnyNDArray = np.array([np.eye(M) for _ in range(N)])
    return 0.1 * (K_np + np.transpose(K_np, (0, 2, 1))) + beye


@check_shapes(
    "q_mu: [broadcast batch...]",
    "q_sigma: [broadcast batch...]",
    "p_var: [broadcast batch...]",
    "return: []",
)
def compute_kl_1d(q_mu: TensorType, q_sigma: TensorType, p_var: TensorType = 1.0) -> TensorType:
    p_var = tf.ones_like(q_sigma) if p_var is None else p_var
    q_var = tf.square(q_sigma)
    kl = 0.5 * (q_var / p_var + tf.square(q_mu) / p_var - 1 + tf.math.log(p_var / q_var))
    return tf.reduce_sum(kl)


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Datum:
    cs = ShapeChecker().check_shape

    M, N = 5, 4

    mu = cs(rng.randn(M, N), "[M, N]")
    A = cs(rng.randn(M, M), "[M, M]")
    I = cs(np.eye(M), "[M, M]")
    K = cs(A @ A.T + default_jitter() * I, "[M, M]")
    sqrt = cs(make_sqrt(N, M), "[N, M, M]")
    sqrt_diag = cs(rng.randn(M, N), "[M, N]")
    K_batch = cs(make_K_batch(N, M), "[N, M, M]")
    K_cholesky = cs(np.linalg.cholesky(K), "[M, M]")


@pytest.mark.parametrize("diag", [True, False])
def test_kl_k_cholesky(diag: bool) -> None:
    """
    Test that passing K or K_cholesky yield the same answer
    """
    q_mu = Datum.mu
    q_sqrt = Datum.sqrt_diag if diag else Datum.sqrt
    kl_K = gauss_kl(q_mu, q_sqrt, K=Datum.K)
    kl_K_chol = gauss_kl(q_mu, q_sqrt, K_cholesky=Datum.K_cholesky)

    np.testing.assert_allclose(kl_K.numpy(), kl_K_chol.numpy())


@pytest.mark.parametrize("white", [True, False])
def test_diags(white: bool) -> None:
    """
    The covariance of q(x) can be Cholesky matrices or diagonal matrices.
    Here we make sure the behaviours overlap.
    """
    # the chols are diagonal matrices, with the same entries as the diag representation.
    chol_from_diag = tf.stack(
        [tf.linalg.diag(Datum.sqrt_diag[:, i]) for i in range(Datum.N)]  # [N, M, M]
    )
    kl_diag = gauss_kl(Datum.mu, Datum.sqrt_diag, Datum.K if white else None)
    kl_dense = gauss_kl(Datum.mu, chol_from_diag, Datum.K if white else None)

    np.testing.assert_allclose(kl_diag, kl_dense)


@pytest.mark.parametrize("diag", [True, False])
def test_whitened(diag: bool) -> None:
    """
    Check that K=Identity and K=None give same answer
    """
    chol_from_diag = tf.stack(
        [tf.linalg.diag(Datum.sqrt_diag[:, i]) for i in range(Datum.N)]  # [N, M, M]
    )
    s = Datum.sqrt_diag if diag else chol_from_diag

    kl_white = gauss_kl(Datum.mu, s)
    kl_nonwhite = gauss_kl(Datum.mu, s, Datum.I)

    np.testing.assert_allclose(kl_white, kl_nonwhite)


@pytest.mark.parametrize("shared_k", [True, False])
@pytest.mark.parametrize("diag", [True, False])
def test_sumkl_equals_batchkl(shared_k: bool, diag: bool) -> None:
    """
    gauss_kl implicitely performs a sum of KL divergences
    This test checks that doing the sum outside of the function is equivalent
    For q(X)=prod q(x_l) and p(X)=prod p(x_l), check that sum KL(q(x_l)||p(x_l)) = KL(q(X)||p(X))
    Here, q(X) has covariance [L, M, M]
    p(X) has covariance [L, M, M] ( or [M, M] )
    Here, q(x_i) has covariance [1, M, M]
    p(x_i) has covariance [M, M]
    """
    s = Datum.sqrt_diag if diag else Datum.sqrt
    kl_batch = gauss_kl(Datum.mu, s, Datum.K if shared_k else Datum.K_batch)
    kl_sum = []
    for n in range(Datum.N):
        q_mu_n = Datum.mu[:, n][:, None]  # [M, 1]
        q_sqrt_n = (
            Datum.sqrt_diag[:, n][:, None] if diag else Datum.sqrt[n, :, :][None, :, :]
        )  # [1, M, M] or [M, 1]
        K_n = Datum.K if shared_k else Datum.K_batch[n, :, :][None, :, :]  # [1, M, M] or [M, M]
        kl_n = gauss_kl(q_mu_n, q_sqrt_n, K=K_n)
        kl_sum.append(kl_n)

    kl_sum = tf.reduce_sum(kl_sum)
    assert_almost_equal(kl_sum, kl_batch)


@patch("tensorflow.__version__", "2.1.0")
def test_sumkl_equals_batchkl_shared_k_not_diag_mocked_tf21() -> None:
    """
    Version of test_sumkl_equals_batchkl with shared_k=True and diag=False
    that tests the TensorFlow < 2.2 workaround with tiling still works.
    """
    kl_batch = gauss_kl(Datum.mu, Datum.sqrt, Datum.K)
    kl_sum = []
    for n in range(Datum.N):
        q_mu_n = Datum.mu[:, n][:, None]  # [M, 1]
        q_sqrt_n = Datum.sqrt[n, :, :][None, :, :]  # [1, M, M] or [M, 1]
        K_n = Datum.K  # [1, M, M] or [M, M]
        kl_n = gauss_kl(q_mu_n, q_sqrt_n, K=K_n)
        kl_sum.append(kl_n)

    kl_sum = tf.reduce_sum(kl_sum)
    assert_almost_equal(kl_sum, kl_batch)


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("white", [True, False])
def test_oned(white: bool, dim: bool) -> None:
    """
    Check that the KL divergence matches a 1D by-hand calculation.
    """
    mu1d = Datum.mu[dim, :][None, :]  # [1, N]
    s1d = Datum.sqrt[:, dim, dim][:, None, None]  # [N, 1, 1]
    K1d = Datum.K_batch[:, dim, dim][:, None, None]  # [N, 1, 1]

    kl = gauss_kl(mu1d, s1d, K1d if not white else None)
    kl_1d = compute_kl_1d(
        tf.reshape(mu1d, (-1,)),  # N
        tf.reshape(s1d, (-1,)),  # N
        None if white else tf.reshape(K1d, (-1,)),
    )  # N
    np.testing.assert_allclose(kl, kl_1d)


def test_unknown_size_inputs() -> None:
    """
    Test for #725 and #734. When the shape of the Gaussian's mean had at least
    one unknown parameter, `gauss_kl` would blow up. This happened because
    `tf.size` can only output types `tf.int32` or `tf.int64`.
    """
    mu: AnyNDArray = np.ones([1, 4], dtype=default_float())
    sqrt: AnyNDArray = np.ones([4, 1, 1], dtype=default_float())

    known_shape = gauss_kl(*map(tf.constant, [mu, sqrt]))
    unknown_shape = gauss_kl(mu, sqrt)

    np.testing.assert_allclose(known_shape, unknown_shape)


@pytest.mark.parametrize("white", [True, False])
def test_q_sqrt_constraints(
    inducing_points: InducingPoints, kernel: Kernel, mu: AnyNDArray, white: bool
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

    kls = []
    for q_sqrt in [q_sqrt_constrained, q_sqrt_unconstrained]:

        with tf.GradientTape() as tape:
            kl = prior_kl(inducing_points, kernel, mu, q_sqrt, whiten=white)

        grad = tape.gradient(kl, q_sqrt.unconstrained_variable)
        q_sqrt.unconstrained_variable.assign_sub(grad)
        kls.append(kl)

    diff_kls_before_gradient_step = kls[0] - kls[1]

    assert_allclose(diff_kls_before_gradient_step, 0)

    diff_after_gradient_step = (q_sqrt_constrained - q_sqrt_unconstrained).numpy()
    assert_allclose(diff_after_gradient_step, 0)
