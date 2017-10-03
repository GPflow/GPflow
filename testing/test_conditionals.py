from __future__ import print_function
import gpflow
import tensorflow as tf
import numpy as np
import unittest

from testing.gpflow_testcase import GPflowTestCase
from gpflow import settings

float_type = settings.dtypes.float_type

class DiagsTest(GPflowTestCase):
    """
    The conditionals can take cholesky matrices or diagonal matrices.

    Here we make sure the behaviours overlap.
    """
    def setUp(self):
        with self.test_session():
            self.num_latent = 2
            self.num_data = 3
            self.k = gpflow.kernels.Matern32(1) + gpflow.kernels.White(1)
            self.k.white.variance = 0.01
            self.X = tf.placeholder(float_type)
            self.mu = tf.placeholder(float_type)
            self.Xs = tf.placeholder(float_type)
            self.sqrt = tf.placeholder(float_type, [self.num_data, self.num_latent])

            # make tf array shenanigans
            self.free_x = tf.placeholder(float_type)
            self.k.make_tf_array(self.free_x)

            self.free_x_data = self.k.get_free_state()
            # NB. with too many random data, numerics suffer
            self.rng = np.random.RandomState(0)
            self.X_data = self.rng.randn(self.num_data,1)
            self.mu_data = self.rng.randn(self.num_data,self.num_latent)
            self.sqrt_data = self.rng.randn(self.num_data,self.num_latent)
            self.Xs_data = self.rng.randn(50,1)

            self.feed_dict = {
                self.X:self.X_data,
                self.Xs:self.Xs_data,
                self.mu:self.mu_data,
                self.sqrt:self.sqrt_data,
                self.free_x:self.free_x_data}

            # the chols are diagonal matrices, with the same entries as the diag representation.
            self.chol = tf.stack([tf.diag(self.sqrt[:,i]) for i in range(self.num_latent)])
            self.chol = tf.transpose(self.chol, perm=[1,2,0])

    def test_whiten(self):
        with self.test_session() as sess, self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = gpflow.conditionals.conditional(
                self.Xs, self.X, self.k, self.mu, q_sqrt=self.sqrt, whiten=True)
            Fstar_mean_2, Fstar_var_2 = gpflow.conditionals.conditional(
                self.Xs, self.X, self.k, self.mu, q_sqrt=self.chol, whiten=True)

            mean1, var1 = sess.run([Fstar_mean_1, Fstar_var_1], feed_dict=self.feed_dict)
            mean2, var2 = sess.run([Fstar_mean_2, Fstar_var_2], feed_dict=self.feed_dict)

            np.testing.assert_allclose(mean1, mean2)
            np.testing.assert_allclose(var1, var2)

    def test_nonwhiten(self):
        with self.test_session() as sess, self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = gpflow.conditionals.conditional(self.Xs, self.X, self.k, self.mu, q_sqrt=self.sqrt)
            Fstar_mean_2, Fstar_var_2 = gpflow.conditionals.conditional(self.Xs, self.X, self.k, self.mu, q_sqrt=self.chol)

            mean1, var1 = sess.run([Fstar_mean_1, Fstar_var_1], feed_dict=self.feed_dict)
            mean2, var2 = sess.run([Fstar_mean_2, Fstar_var_2], feed_dict=self.feed_dict)

            np.testing.assert_allclose(mean1, mean2)
            np.testing.assert_allclose(var1, var2)


class WhitenTest(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.k = gpflow.kernels.Matern32(1) + gpflow.kernels.White(1)
            self.k.white.variance = 0.01
            self.num_data = 10
            self.num_test_data = 100
            self.X = tf.placeholder(float_type, [self.num_data, 1])
            self.F = tf.placeholder(float_type, [self.num_data, 1])
            self.Xs = tf.placeholder(float_type, [self.num_test_data, 1])

            # make tf array shenanigans
            self.free_x = tf.placeholder(float_type)
            self.k.make_tf_array(self.free_x)

            self.free_x_data = self.k.get_free_state()
            # NB. with too many random data, numerics suffer
            self.rng = np.random.RandomState(0)
            self.X_data = self.rng.randn(self.num_data, 1)
            self.F_data = self.rng.randn(self.num_data, 1)
            self.Xs_data = self.rng.randn(self.num_test_data,1)

            self.feed_dict = {
                    self.free_x:self.free_x_data,
                    self.X:self.X_data,
                    self.F:self.F_data,
                    self.Xs:self.Xs_data}

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        same as the non-whitened one.
        """

        with self.test_session() as sess, self.k.tf_mode():
            K = self.k.K(self.X) + tf.eye(self.num_data, dtype=float_type) * 1e-6
            L = tf.cholesky(K)
            V = tf.matrix_triangular_solve(L, self.F, lower=True)
            Fstar_mean, Fstar_var = gpflow.conditionals.conditional(self.Xs, self.X, self.k, self.F)
            Fstar_w_mean, Fstar_w_var = gpflow.conditionals.conditional(self.Xs, self.X, self.k, V, whiten=True)

            mean1, var1 = sess.run([Fstar_w_mean, Fstar_w_var], feed_dict=self.feed_dict)
            mean2, var2 = sess.run([Fstar_mean, Fstar_var], feed_dict=self.feed_dict)

            np.testing.assert_allclose(mean1, mean2, 1e-6, 1e-6)  # TODO: should tolerance be type dependent?
            np.testing.assert_allclose(var1, var2, 1e-6, 1e-6)


class WhitenTestGaussian(WhitenTest):
    def setUp(self):
        WhitenTest.setUp(self)
        with self.test_session() as sess, self.k.tf_mode():
            self.F_sqrt = tf.placeholder(float_type, [self.num_data, 1])
            self.F_sqrt_data = self.rng.rand(self.num_data,1)
            self.feed_dict[self.F_sqrt] = self.F_sqrt_data

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        same as the non-whitened one.
        """
        with self.test_session() as sess, self.k.tf_mode():
            K = self.k.K(self.X)
            L = tf.cholesky(K)
            V = tf.matrix_triangular_solve(L, self.F, lower=True)
            V_chol = tf.matrix_triangular_solve(L, tf.diag(self.F_sqrt[:,0]), lower=True)
            V_sqrt = tf.expand_dims(V_chol, 2)

            Fstar_mean, Fstar_var = gpflow.conditionals.conditional(self.Xs, self.X, self.k, self.F, q_sqrt=self.F_sqrt)
            Fstar_w_mean, Fstar_w_var = gpflow.conditionals.conditional(self.Xs, self.X, self.k, V, q_sqrt=V_sqrt, whiten=True)

            mean1, var1 = sess.run([Fstar_w_mean, Fstar_w_var], feed_dict=self.feed_dict)
            mean2, var2 = sess.run([Fstar_mean, Fstar_var], feed_dict=self.feed_dict)

            np.testing.assert_almost_equal(mean1, mean2, 4)
            np.testing.assert_almost_equal(var1, var2, 4)

if __name__ == "__main__":
    unittest.main()
