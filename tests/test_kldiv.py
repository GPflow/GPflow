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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_almost_equal

from gpflow.kullback_leiblers import gauss_kl
from gpflow import default_float, default_jitter

rng = np.random.RandomState(0)

# ------------------------------------------
# Helpers
# ------------------------------------------


def make_sqrt(N, M):
    return np.array([np.tril(rng.randn(M, M)) for _ in range(N)])  # [N, M, M]


def make_K_batch(N, M):
    K_np = rng.randn(N, M, M)
    beye = np.array([np.eye(M) for _ in range(N)])
    return .1 * (K_np + np.transpose(K_np, (0, 2, 1))) + beye


def compute_kl_1d(q_mu, q_sigma, p_var=1.0):
    p_var = tf.ones_like(q_sigma) if p_var is None else p_var
    q_var = tf.square(q_sigma)
    kl = 0.5 * (q_var / p_var + tf.square(q_mu) / p_var - 1 + tf.math.log(p_var / q_var))
    return tf.reduce_sum(kl)


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Datum:
    M, N = 5, 4

    mu = rng.randn(M, N)  # [M, N]
    A = rng.randn(M, M)
    I = np.eye(M)  # [M, M]
    K = A @ A.T + default_jitter() * I  # [M, M]
    sqrt = make_sqrt(N, M)  # [N, M, M]
    sqrt_diag = rng.randn(M, N)  # [M, N]
    K_batch = make_K_batch(N, M)


@pytest.mark.parametrize('white', [True, False])
def test_diags(white):
    """
    The covariance of q(x) can be Cholesky matrices or diagonal matrices.
    Here we make sure the behaviours overlap.
    """
    # the chols are diagonal matrices, with the same entries as the diag representation.
    chol_from_diag = tf.stack([tf.linalg.diag(Datum.sqrt_diag[:, i]) for i in range(Datum.N)]  # [N, M, M]
                              )
    kl_diag = gauss_kl(Datum.mu, Datum.sqrt_diag, Datum.K if white else None)
    kl_dense = gauss_kl(Datum.mu, chol_from_diag, Datum.K if white else None)

    np.testing.assert_allclose(kl_diag, kl_dense)


@pytest.mark.parametrize('diag', [True, False])
def test_whitened(diag):
    """
    Check that K=Identity and K=None give same answer
    """
    chol_from_diag = tf.stack([tf.linalg.diag(Datum.sqrt_diag[:, i]) for i in range(Datum.N)]  # [N, M, M]
                              )
    s = Datum.sqrt_diag if diag else chol_from_diag

    kl_white = gauss_kl(Datum.mu, s)
    kl_nonwhite = gauss_kl(Datum.mu, s, Datum.I)

    np.testing.assert_allclose(kl_white, kl_nonwhite)


@pytest.mark.parametrize('shared_k', [True, False])
@pytest.mark.parametrize('diag', [True, False])
def test_sumkl_equals_batchkl(shared_k, diag):
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
        q_sqrt_n = Datum.sqrt_diag[:, n][:, None] if diag else Datum.sqrt[n, :, :][None, :, :]  # [1, M, M] or [M, 1]
        K_n = Datum.K if shared_k else Datum.K_batch[n, :, :][None, :, :]  # [1, M, M] or [M, M]
        kl_n = gauss_kl(q_mu_n, q_sqrt_n, K=K_n)
        kl_sum.append(kl_n)

    kl_sum = tf.reduce_sum(kl_sum)
    assert_almost_equal(kl_sum, kl_batch)


@pytest.mark.parametrize('dim', [0, 1])
@pytest.mark.parametrize('white', [True, False])
def test_oned(white, dim):
    """
    Check that the KL divergence matches a 1D by-hand calculation.
    """
    mu1d = Datum.mu[dim, :][None, :]  # [1, N]
    s1d = Datum.sqrt[:, dim, dim][:, None, None]  # [N, 1, 1]
    K1d = Datum.K_batch[:, dim, dim][:, None, None]  # [N, 1, 1]

    kl = gauss_kl(mu1d, s1d, K1d if not white else None)
    kl_1d = compute_kl_1d(
        tf.reshape(mu1d, (-1, )),  # N
        tf.reshape(s1d, (-1, )),  # N
        None if white else tf.reshape(K1d, (-1, )))  # N
    np.testing.assert_allclose(kl, kl_1d)


def test_unknown_size_inputs():
    """
    Test for #725 and #734. When the shape of the Gaussian's mean had at least
    one unknown parameter, `gauss_kl` would blow up. This happened because
    `tf.size` can only output types `tf.int32` or `tf.int64`.
    """
    mu = np.ones([1, 4], dtype=default_float())
    sqrt = np.ones([4, 1, 1], dtype=default_float())

    known_shape = gauss_kl(*map(tf.constant, [mu, sqrt]))
    unknown_shape = gauss_kl(mu, sqrt)

    np.testing.assert_allclose(known_shape, unknown_shape)
