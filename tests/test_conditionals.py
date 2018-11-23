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


import pytest
from gpflow.test_util import session_tf
import tensorflow as tf

import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose


import gpflow
from gpflow.test_util import GPflowTestCase
from gpflow import settings


class DiagsTest(GPflowTestCase):
    """
    The conditionals can take cholesky matrices or diagaonal matrices.

    Here we make sure the behaviours overlap.
    """
    def prepare(self):
        num_latent = 2
        num_data = 3
        k = gpflow.kernels.Matern32(1) + gpflow.kernels.White(1)
        k.kernels[1].variance = 0.01
        X = tf.placeholder(settings.float_type)
        mu = tf.placeholder(settings.float_type)
        Xs = tf.placeholder(settings.float_type)
        sqrt = tf.placeholder(settings.float_type, shape=[num_data, num_latent])

        rng = np.random.RandomState(0)
        X_data = rng.randn(num_data, 1)
        mu_data = rng.randn(num_data, num_latent)
        sqrt_data = rng.randn(num_data, num_latent)
        Xs_data = rng.randn(50, 1)

        feed_dict = {X: X_data, Xs: Xs_data, mu: mu_data, sqrt: sqrt_data}
        k.compile()

        #the chols are diagonal matrices, with the same entries as the diag representation.
        chol = tf.stack([tf.diag(sqrt[:, i]) for i in range(num_latent)])
        return Xs, X, k, mu, sqrt, chol, feed_dict

    def test_whiten(self):
        with self.test_context() as sess:
            Xs, X, k, mu, sqrt, chol, feed_dict = self.prepare()

            Fstar_mean_1, Fstar_var_1 = gpflow.conditionals.conditional(
                Xs, X, k, mu, q_sqrt=sqrt)
            Fstar_mean_2, Fstar_var_2 = gpflow.conditionals.conditional(
                Xs, X, k, mu, q_sqrt=chol, white=True)

            mean_diff = sess.run(Fstar_mean_1 - Fstar_mean_2, feed_dict=feed_dict)
            var_diff = sess.run(Fstar_var_1 - Fstar_var_2, feed_dict=feed_dict)

            # TODO(@awav): CHECK IT
            # assert_allclose(mean_diff, 0.0)
            # assert_allclose(var_diff, 0.0)

    def test_nonwhiten(self):
        with self.test_context() as sess:
            Xs, X, k, mu, sqrt, chol, feed_dict = self.prepare()

            Fstar_mean_1, Fstar_var_1 = gpflow.conditionals.conditional(
                Xs, X, k, mu, q_sqrt=sqrt)
            Fstar_mean_2, Fstar_var_2 = gpflow.conditionals.conditional(
                Xs, X, k, mu, q_sqrt=chol)

            mean_diff = sess.run(Fstar_mean_1 - Fstar_mean_2, feed_dict=feed_dict)
            var_diff = sess.run(Fstar_var_1 - Fstar_var_2, feed_dict=feed_dict)

            assert_allclose(mean_diff, 0)
            assert_allclose(var_diff, 0)


class WhitenTest(GPflowTestCase):
    def prepare(self):
        k = gpflow.kernels.Matern32(1) + gpflow.kernels.White(1)
        k.kernels[1].variance = 0.01

        num_data = 10
        num_test_data = 100
        X = tf.placeholder(settings.float_type, [num_data, 1])
        F = tf.placeholder(settings.float_type, [num_data, 1])
        Xs = tf.placeholder(settings.float_type, [num_test_data, 1])

        rng = np.random.RandomState(0)
        X_data = rng.randn(num_data, 1)
        F_data = rng.randn(num_data, 1)
        Xs_data = rng.randn(num_test_data, 1)

        feed_dict = {X: X_data, F: F_data, Xs: Xs_data}

        return Xs, X, F, k, num_data, feed_dict

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one.
        """

        with self.test_context() as sess:
            Xs, X, F, k, num_data, feed_dict = self.prepare()
            k.compile(session=sess)

            K = k.K(X) + tf.eye(num_data, dtype=settings.float_type) * 1e-6
            L = tf.cholesky(K)
            V = tf.matrix_triangular_solve(L, F, lower=True)
            Fstar_mean, Fstar_var = gpflow.conditionals.conditional(Xs, X, k, F)
            Fstar_w_mean, Fstar_w_var = gpflow.conditionals.conditional(Xs, X, k, V, white=True)

            mean1, var1 = sess.run([Fstar_w_mean, Fstar_w_var], feed_dict=feed_dict)
            mean2, var2 = sess.run([Fstar_mean, Fstar_var], feed_dict=feed_dict)

             # TODO: should tolerance be type dependent?
            assert_allclose(mean1, mean2)
            assert_allclose(var1, var2)


class WhitenTestGaussian(WhitenTest):
    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one.
        """
        with self.test_context() as sess:
            rng = np.random.RandomState(0)
            Xs, X, F, k, num_data, feed_dict = self.prepare()
            k.compile(session=sess)

            F_sqrt = tf.placeholder(settings.float_type, [num_data, 1])
            F_sqrt_data = rng.rand(num_data, 1)
            feed_dict[F_sqrt] = F_sqrt_data

            K = k.K(X)
            L = tf.cholesky(K)
            V = tf.matrix_triangular_solve(L, F, lower=True)
            V_sqrt = tf.matrix_triangular_solve(L, tf.diag(F_sqrt[:, 0]), lower=True)[None, :, :]

            Fstar_mean, Fstar_var = gpflow.conditionals.conditional(
                Xs, X, k, F, q_sqrt=F_sqrt)
            Fstar_w_mean, Fstar_w_var = gpflow.conditionals.conditional(
                Xs, X, k, V, q_sqrt=V_sqrt, white=True)

            mean_difference = sess.run(Fstar_w_mean - Fstar_mean, feed_dict=feed_dict)
            var_difference = sess.run(Fstar_w_var - Fstar_var, feed_dict=feed_dict)

            assert_allclose(mean_difference, 0, atol=4)
            assert_allclose(var_difference, 0, atol=4)



@pytest.mark.parametrize("full_cov", [True, False])
def test_vs_ref(session_tf, full_cov):
    """
    Test that conditionals agree with a slow-but-clear numpy implementation
    """
    Dy, N, M, Dx = 6, 4, 3, 2
    X = np.random.randn(N, Dx)
    Z = np.random.randn(M, Dx)
    kern = gpflow.kernels.Matern52(Dx, lengthscales=0.5)

    Kmm = kern.compute_K_symm(Z) + np.eye(M) * settings.numerics.jitter_level
    Kmn = kern.compute_K(Z, X)
    Knn = kern.compute_K_symm(X)

    Kmm, Kmn, Knm, Knn = [np.tile(k[None, :, :], [Dy, 1, 1]) for k in [Kmm, Kmn, Kmn.T, Knn]]

    q_mu = np.random.randn(M, Dy)
    q_sqrt = np.tril(np.random.randn(Dy, M, M), -1)

    S = q_sqrt @ np.transpose(q_sqrt, [0, 2, 1])

    Kmm_inv = np.linalg.inv(Kmm)
    mean_np = np.einsum('dmn,dmM,Md->nd', Kmn, Kmm_inv, q_mu)
    cov_np = Knn + Knm @ Kmm_inv @ (S - Kmm) @ Kmm_inv @ Kmn

    mean_tf, cov_tf = gpflow.conditionals.conditional(X, Z, kern, q_mu,
                                                      q_sqrt=tf.identity(q_sqrt),
                                                      white=False,
                                                      full_cov=full_cov)

    mean_tf, cov_tf = session_tf.run([mean_tf, cov_tf])

    if not full_cov:
        cov_np = np.diagonal(cov_np, axis1=-1, axis2=-2).T

    assert_allclose(mean_np, mean_tf)
    assert_allclose(cov_np, cov_tf)


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("white", [True, False])
def test_multisample2(session_tf, full_cov, white):
    """
    Test that the conditional broadcasts correctly over leading dimensions of Xnew

    Xnew can be shape [...,N,D], and conditional should broadcast over the [...]

    """
    S1, S2, Dy, N, M, Dx = 7, 6, 5, 4, 3, 2

    SX = np.random.randn(S1*S2, N, Dx)
    S1_S2_X = np.reshape(SX, [S1, S2, N, Dx])
    Z = np.random.randn(M, Dx)
    kern = gpflow.kernels.Matern52(Dx, lengthscales=0.5)

    q_mu = np.random.randn(M, Dy)
    q_sqrt = np.tril(np.random.randn(Dy, M, M), -1)

    x = tf.placeholder(tf.float64, [None, None])

    mean_tf, cov_tf = gpflow.conditionals.conditional(x, Z, kern, q_mu,
                                                      q_sqrt=tf.identity(q_sqrt),
                                                      white=white,
                                                      full_cov=full_cov)
    ms, vs = [], []
    for X in SX:
        m, v = session_tf.run([mean_tf, cov_tf], {x:X})
        ms.append(m)
        vs.append(v)

    ms = np.array(ms)
    vs = np.array(vs)

    ms_S12, vs_S12 = session_tf.run(gpflow.conditionals.conditional(SX, Z, kern, q_mu,
                                                                    q_sqrt=tf.identity(q_sqrt),
                                                                    white=white,
                                                                    full_cov=full_cov))

    ms_S1_S2, vs_S1_S2 = session_tf.run(gpflow.conditionals.conditional(S1_S2_X, Z, kern, q_mu,
                                                                        q_sqrt=tf.identity(q_sqrt),
                                                                        white=white,
                                                                        full_cov=full_cov))

    assert_allclose(ms_S12, ms)
    assert_allclose(vs_S12, vs)

    assert_allclose(ms_S1_S2.reshape(S1 * S2, N, Dy), ms)

    if full_cov:
        assert_allclose(vs_S1_S2.reshape(S1 * S2, Dy, N, N), vs)
    else:
        assert_allclose(vs_S1_S2.reshape(S1 * S2, N, Dy), vs)


if __name__ == '__main__':
    tf.test.main()
