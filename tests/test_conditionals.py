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


if __name__ == '__main__':
    tf.test.main()
