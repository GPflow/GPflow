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
# limitations under the License.from __future__ import print_function

# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase


def squareT(A):
    """
    Returns (A Aᵀ)
    """
    return A.dot(A.T)


class DiagsTest(GPflowTestCase):
    """
    The covariance of q(x) can be Cholesky matrices or diagonal matrices.

    Here we make sure the behaviours overlap.
    """

    def setUp(self):
        with self.test_session():
            N = 4
            M = 5
            self.mu = tf.placeholder(settings.float_type, [M, N])
            self.sqrt = tf.placeholder(settings.float_type, [M, N])
            self.K = tf.placeholder(settings.float_type, [M, M])

            self.rng = np.random.RandomState(0)
            self.mu_data = self.rng.randn(M, N)
            self.sqrt_data = self.rng.randn(M, N)
            Ksqrt = self.rng.randn(M, M)
            self.K_data = squareT(Ksqrt) + 1e-6 * np.eye(M)

            self.feed_dict = {
                self.mu: self.mu_data,
                self.sqrt: self.sqrt_data,
                self.K: self.K_data,
            }

            # the chols are diagonal matrices, with the same entries as the diag representation.
            self.chol = tf.stack([tf.diag(self.sqrt[:, i]) for i in range(N)])

    def test_white(self):
        with self.test_session() as sess:
            kl_diag = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt)
            kl_dense = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol)

            res_diag = sess.run(kl_diag, feed_dict=self.feed_dict)
            res_dense = sess.run(kl_dense, feed_dict=self.feed_dict)

            np.testing.assert_allclose(res_diag, res_dense)

    def test_nonwhite(self):
        with self.test_session() as sess:
            kl_diag = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt, self.K)
            kl_dense = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol, self.K)

            res_diag = sess.run(kl_diag, feed_dict=self.feed_dict)
            res_dense = sess.run(kl_dense, feed_dict=self.feed_dict)

            np.testing.assert_allclose(res_diag, res_dense)


class WhitenedTest(GPflowTestCase):
    """
    Check that K=Identity and K=None give same answer
    """

    def setUp(self):
        with self.test_session():
            N = 4
            M = 5
            self.mu = tf.placeholder(settings.float_type, [M, N])
            self.sqrt = tf.placeholder(settings.float_type, [M, N])
            self.chol = tf.placeholder(settings.float_type, [M, M, N])
            self.I = tf.placeholder(settings.float_type, [M, M])

            self.rng = np.random.RandomState(0)
            self.mu_data = self.rng.randn(M, N)
            self.sqrt_data = self.rng.randn(M, N)
            q_sqrt = np.rollaxis(np.array([np.tril(self.rng.randn(M, M)) for _ in range(N)]),
                                 0, 3)
            self.chol_data = q_sqrt

            self.feed_dict = {
                self.mu: self.mu_data,
                self.sqrt: self.sqrt_data,
                self.chol: self.chol_data,
                self.I: np.eye(M),
            }

    def test_diag(self):
        with self.test_session() as sess:
            kl_white = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt)
            kl_nonwhite = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt, self.I)

            res_white = sess.run(kl_white, feed_dict=self.feed_dict)
            res_nonwhite = sess.run(kl_nonwhite, feed_dict=self.feed_dict)

            np.testing.assert_allclose(res_white, res_nonwhite)

    def test_dense(self):
        with self.test_session() as sess:
            kl_white = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol)
            kl_nonwhite = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol, self.I)

            res_white = sess.run(kl_white, feed_dict=self.feed_dict)
            res_nonwhite = sess.run(kl_nonwhite, feed_dict=self.feed_dict)

            np.testing.assert_allclose(res_white, res_nonwhite)


class EqualityTest(GPflowTestCase):
    """
    Check that the KL divergence is zero if q == p.
    """

    def setUp(self):
        with self.test_session():
            N = 4
            M = 5
            self.mu = tf.placeholder(settings.float_type, [M, N])
            self.sqrt = tf.placeholder(settings.float_type, [M, N])
            self.chol = tf.placeholder(settings.float_type, [N, M, M])
            self.K = tf.placeholder(settings.float_type, [M, M])
            self.Kdiag = tf.placeholder(settings.float_type, [M, M])

            self.rng = np.random.RandomState(0)
            self.mu_data = self.rng.randn(M, N)
            sqrt_diag = self.rng.randn(M)
            self.sqrt_data = np.array([sqrt_diag for _ in range(N)]).T
            sqrt_chol = np.tril(self.rng.randn(M, M))
            self.chol_data = np.array([sqrt_chol for _ in range(N)])

            self.feed_dict = {
                self.mu: np.zeros((M, N)),
                self.sqrt: self.sqrt_data,
                self.chol: self.chol_data,
                self.K: squareT(sqrt_chol),
                self.Kdiag: np.diag(sqrt_diag ** 2),
            }

    def test_diag(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt, self.Kdiag)
            res = sess.run(kl, feed_dict=self.feed_dict)
            self.assertTrue(np.allclose(res, 0.0))

    def test_dense(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol, self.K)
            res = sess.run(kl, feed_dict=self.feed_dict)
            self.assertTrue(np.allclose(res, 0.0))


def np_kl_1d(q_mu, q_sigma, p_var=1.0):
    q_var = q_sigma ** 2
    return 0.5 * (q_var / p_var + q_mu ** 2 / p_var - 1 + np.log(p_var / q_var))


def np_kl_1d_many(q_mus, q_sigmas, p_var=1.0):
    kls = [np_kl_1d(q_mu, q_sigma, p_var) for q_mu, q_sigma in zip(q_mus, q_sigmas)]
    return np.sum(kls)


class OneDTest(GPflowTestCase):
    """
    Check that the KL divergence matches a 1D by-hand calculation.
    """

    def setUp(self):
        with self.test_session():
            N = 2
            M = 1
            self.mu = tf.placeholder(settings.float_type, [M, N])
            self.sqrt = tf.placeholder(settings.float_type, [M, N])
            self.chol = tf.placeholder(settings.float_type, [N, M, M])
            self.K = tf.placeholder(settings.float_type, [M, M])
            self.Kdiag = tf.placeholder(settings.float_type, [M, M])

            self.mu_data = np.array([[1.3], [1.7]]).T
            self.sqrt_data = np.array([[0.8], [1.5]]).T
            self.chol_data = self.sqrt_data.T[:, :, None]
            self.K_data = np.array([[2.5]])

            self.feed_dict = {
                self.mu: self.mu_data,
                self.sqrt: self.sqrt_data,
                self.chol: self.chol_data,
                self.K: self.K_data,
            }

    def test_diag_white(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt)
            res = sess.run(kl, feed_dict=self.feed_dict)
            np_kl = np_kl_1d_many(self.mu_data[0, :], self.sqrt_data[0, :])
            np.testing.assert_allclose(res, np_kl)

    def test_diag_nonwhite(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.sqrt, self.K)
            res = sess.run(kl, feed_dict=self.feed_dict)
            np_kl = np_kl_1d_many(self.mu_data[0, :], self.sqrt_data[0, :], self.K_data[0, 0])
            np.testing.assert_allclose(res, np_kl)

    def test_dense_white(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol)
            res = sess.run(kl, feed_dict=self.feed_dict)
            np_kl = np_kl_1d_many(self.mu_data[0, :], self.sqrt_data[0, :])
            np.testing.assert_allclose(res, np_kl)

    def test_dense_nonwhite(self):
        with self.test_session() as sess:
            kl = gpflow.kullback_leiblers.gauss_kl(self.mu, self.chol, self.K)
            res = sess.run(kl, feed_dict=self.feed_dict)
            np_kl = np_kl_1d_many(self.mu_data[0, :], self.sqrt_data[0, :], self.K_data[0, 0])
            np.testing.assert_allclose(res, np_kl)


if __name__ == "__main__":
    unittest.main()
