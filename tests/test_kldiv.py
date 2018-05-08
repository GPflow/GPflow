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
import tensorflow as tf

import gpflow
from gpflow.kullback_leiblers import gauss_kl
from numpy.testing import assert_almost_equal
import pytest
from gpflow import settings
from gpflow.test_util import session_tf

def squareT(A):
    """
    Returns (A Aᵀ)
    """
    return A.dot(A.T)

def make_sqrt_data(rng, N, M):
    return np.array([np.tril(rng.randn(M, M)) for _ in range(N)]) # N x M x M

def make_K_batch_data(rng, N, M):
    K_np = rng.randn(N, M, M)
    beye = np.array([np.eye(M) for _ in range(N)])
    return .1 * (K_np + np.transpose(K_np, (0, 2, 1))) + beye

class Datum:
    M, N = 5, 4
    rng = np.random.RandomState(0)
    mu_data = rng.randn(M, N)  # M x N
    K_data = squareT(rng.randn(M, M)) + 1e-6 * np.eye(M)  # M x M
    I = np.eye(M) # M x M
    sqrt_data = make_sqrt_data(rng, N, M) # N x M x M
    sqrt_diag_data = rng.randn(M, N) # M x N
    K_batch_data = make_K_batch_data(rng, N, M)

@pytest.fixture
def mu(session_tf):
    return tf.convert_to_tensor(Datum.mu_data)

@pytest.fixture
def sqrt_diag(session_tf):
    return tf.convert_to_tensor(Datum.sqrt_diag_data)

@pytest.fixture
def K(session_tf):
    return tf.convert_to_tensor(Datum.K_data)

@pytest.fixture
def K_batch(session_tf):
    return tf.convert_to_tensor(Datum.K_batch_data)

@pytest.fixture
def sqrt(session_tf):
    return tf.convert_to_tensor(Datum.sqrt_data)

@pytest.fixture()
def I(session_tf):
    return tf.convert_to_tensor(Datum.I)

@pytest.mark.parametrize('white', [True, False])
def test_diags(session_tf, white, mu, sqrt_diag, K):
    """
    The covariance of q(x) can be Cholesky matrices or diagonal matrices.
    Here we make sure the behaviours overlap.
    """
    # the chols are diagonal matrices, with the same entries as the diag representation.
    chol_from_diag = tf.stack([tf.diag(sqrt_diag[:, i]) for i in range(Datum.N)]) # N x M x M
    # run
    kl_diag = gauss_kl(mu, sqrt_diag, K if white else None)
    kl_dense = gauss_kl(mu, chol_from_diag, K if white else None)

    np.testing.assert_allclose(kl_diag.eval(), kl_dense.eval())

@pytest.mark.parametrize('diag', [True, False])
def test_whitened(session_tf, diag, mu, sqrt_diag, I):
    """
    Check that K=Identity and K=None give same answer
    """
    chol_from_diag = tf.stack([tf.diag(sqrt_diag[:, i]) for i in range(Datum.N)]) # N x M x M
    s = sqrt_diag if diag else chol_from_diag

    kl_white = gauss_kl(mu, s)
    kl_nonwhite = gauss_kl(mu, s, I)

    np.testing.assert_allclose(kl_white.eval(), kl_nonwhite.eval())

@pytest.mark.parametrize('shared_k', [True, False])
@pytest.mark.parametrize('diag', [True, False])
def test_sumkl_equals_batchkl(session_tf, shared_k, diag, mu,
                              sqrt, sqrt_diag, K_batch, K):
    """
    gauss_kl implicitely performs a sum of KL divergences
    This test checks that doing the sum outside of the function is equivalent
    For q(X)=prod q(x_l) and p(X)=prod p(x_l), check that sum KL(q(x_l)||p(x_l)) = KL(q(X)||p(X))
    Here, q(X) has covariance L x M x M
    p(X) has covariance L x M x M ( or M x M )
    Here, q(x_i) has covariance 1 x M x M
    p(x_i) has covariance M x M
    """
    s = sqrt_diag if diag else sqrt
    kl_batch = gauss_kl(mu,s,K if shared_k else K_batch)
    kl_sum = []
    for n in range(Datum.N):
        kl_sum.append(gauss_kl(mu[:, n][:,None], # M x 1
            sqrt_diag[:, n][:, None] if diag else sqrt[n, :, :][None, :, :], # 1 x M x M or M x 1
            K if shared_k else K_batch[n, :, :][None,:,:])) # 1 x M x M or M x M
    kl_sum =tf.reduce_sum(kl_sum)
    assert_almost_equal(kl_sum.eval(), kl_batch.eval())

def tf_kl_1d(q_mu, q_sigma, p_var=1.0):
    p_var = tf.ones_like(q_sigma) if p_var is None else p_var
    q_var = tf.square(q_sigma)
    kl = 0.5 * (q_var / p_var + tf.square(q_mu) / p_var - 1 + tf.log(p_var / q_var))
    return tf.reduce_sum(kl)

@pytest.mark.parametrize('white', [True, False])
def test_oned(session_tf, white, mu, sqrt, K_batch):
    """
    Check that the KL divergence matches a 1D by-hand calculation.
    """
    m = 0
    mu1d = mu[m,:][None,:] # 1 x N
    s1d = sqrt[:,m,m][:,None,None] # N x 1 x 1
    K1d = K_batch[:,m,m][:,None,None] # N x 1 x 1

    kl = gauss_kl(mu1d,s1d,K1d if not white else None)
    kl_tf = tf_kl_1d(tf.reshape(mu1d,(-1,)), # N
                   tf.reshape(s1d,(-1,)), # N
                   None if white else tf.reshape(K1d,(-1,))) # N
    np.testing.assert_allclose(kl.eval(), kl_tf.eval())


def test_unknown_size_inputs(session_tf):
    """
    Test for #725 and #734. When the shape of the Gaussian's mean had at least
    one unknown parameter, `gauss_kl` would blow up. This happened because
    `tf.size` can only output types `tf.int32` or `tf.int64`.
    """
    mu_ph = tf.placeholder(settings.float_type, [None, None])
    sqrt_ph = tf.placeholder(settings.float_type, [None, None, None])
    mu = np.ones([1, 4], dtype=settings.float_type)
    sqrt = np.ones([4, 1, 1], dtype=settings.float_type)
    
    feed_dict = {mu_ph: mu, sqrt_ph: sqrt}
    known_shape_tf = gauss_kl(*map(tf.constant, [mu, sqrt]))
    unknown_shape_tf = gauss_kl(mu_ph, sqrt_ph)
    
    known_shape = session_tf.run(known_shape_tf)
    unknown_shape = session_tf.run(unknown_shape_tf, feed_dict=feed_dict)
    
    np.testing.assert_allclose(known_shape, unknown_shape)


if __name__ == "__main__":
    tf.test.main()
