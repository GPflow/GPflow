import GPflow
import tensorflow as tf
from GPflow.tf_hacks import eye
import numpy as np
import unittest

class DiagsTest(unittest.TestCase):
    """
    The conditionals can take cholesky matrices or diagaonal matrices. 

    Here we make sure the behaviours overlap.
    """
    def setUp(self):
        self.num_latent = 2
        self.k = GPflow.kernels.Matern32(1) + GPflow.kernels.White(1)
        self.k.k2.variance = 0.01
        self.X = tf.placeholder('float64')
        self.mu = tf.placeholder('float64')
        self.Xs = tf.placeholder('float64')
        self.sqrt = tf.placeholder('float64', shape=[3, self.num_latent])

        #make tf array shenanigans
        self.free_x = tf.placeholder('float64')
        self.k.make_tf_array(self.free_x)

        self.free_x_data = self.k.get_free_state()
        # NB. with too many random data, numerics suffer
        self.rng = np.random.RandomState(0)
        self.X_data = self.rng.randn(3,1)
        self.mu_data = self.rng.randn(3,self.num_latent)
        self.sqrt_data = self.rng.randn(3,self.num_latent)
        self.Xs_data = self.rng.randn(50,1)

        self.feed_dict = {
            self.X:self.X_data,
            self.Xs:self.Xs_data,
            self.mu:self.mu_data,
            self.sqrt:self.sqrt_data,
            self.free_x:self.free_x_data}


        #the chols are diagonal matrices, with the same entries as the diag representation.
        self.chol = tf.pack([tf.diag(self.sqrt[:,i]) for i in range(self.num_latent)])
        self.chol = tf.transpose(self.chol, perm=[1,2,0])

    def test_whiten(self):
        with self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, self.mu, self.sqrt, self.num_latent)
            Fstar_mean_2, Fstar_var_2 = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, self.mu, self.chol, self.num_latent)
            
        mean_diff = tf.Session().run(Fstar_mean_1 - Fstar_mean_2, feed_dict=self.feed_dict)
        var_diff = tf.Session().run(Fstar_var_1 - Fstar_var_2, feed_dict=self.feed_dict)

        self.failUnless(np.allclose(mean_diff, 0))
        self.failUnless(np.allclose(var_diff, 0))


    def test_nonwhiten(self):
        with self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.mu, self.sqrt, self.num_latent)
            Fstar_mean_2, Fstar_var_2 = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.mu, self.chol, self.num_latent)
            
        mean_diff = tf.Session().run(Fstar_mean_1 - Fstar_mean_2, feed_dict=self.feed_dict)
        var_diff = tf.Session().run(Fstar_var_1 - Fstar_var_2, feed_dict=self.feed_dict)

        self.failUnless(np.allclose(mean_diff, 0))
        self.failUnless(np.allclose(var_diff, 0))


class WhitenTest(unittest.TestCase):
    def setUp(self):
        self.k = GPflow.kernels.Matern32(1) + GPflow.kernels.White(1)
        self.k.k2.variance = 0.01
        self.X = tf.placeholder('float64')
        self.F = tf.placeholder('float64')
        self.Xs = tf.placeholder('float64')

        #make tf array shenanigans
        self.free_x = tf.placeholder('float64')
        self.k.make_tf_array(self.free_x)

        self.free_x_data = self.k.get_free_state()
        # NB. with too many random data, numerics suffer
        self.rng = np.random.RandomState(0)
        self.X_data = self.rng.randn(3,1)
        self.F_data = self.rng.randn(3,1)
        self.Xs_data = self.rng.randn(50,1)

        self.feed_dict = {
                self.free_x:self.free_x_data,
                self.X:self.X_data,
                self.F:self.F_data,
                self.Xs:self.Xs_data}

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one. 
        """
        
        with self.k.tf_mode():
            K = self.k.K(self.X)
            L = tf.cholesky(K)
            V = tf.user_ops.triangular_solve(L, self.F, 'lower')
            Fstar_mean, Fstar_var = GPflow.conditionals.gp_predict(self.Xs, self.X, self.k, self.F)
            Fstar_w_mean, Fstar_w_var = GPflow.conditionals.gp_predict_whitened(self.Xs, self.X, self.k, V)


        mean_difference = tf.Session().run(Fstar_w_mean - Fstar_mean, feed_dict=self.feed_dict)
        var_difference = tf.Session().run(Fstar_w_var - Fstar_var, feed_dict=self.feed_dict)

        self.failUnless(np.all(np.abs(mean_difference) < 1e-2))
        self.failUnless(np.all(np.abs(var_difference) < 1e-2))


class WhitenTestGaussian(WhitenTest):
    def setUp(self):
        WhitenTest.setUp(self)
        self.F_sqrt = tf.placeholder('float64')
        self.F_sqrt_data = self.rng.randn(3,1)
        self.feed_dict[self.F_sqrt] = self.F_sqrt_data

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one. 
        """
        with self.k.tf_mode():
            K = self.k.K(self.X)
            L = tf.cholesky(K)
            V = tf.user_ops.triangular_solve(L, self.F, 'lower')
            Li = tf.user_ops.triangular_solve(L, eye(3), 'lower')
            V_var = tf.matmul( tf.matmul(Li, tf.diag(tf.square(tf.reshape(self.F_sqrt, (-1,))))), tf.transpose(Li))
            V_sqrt = tf.expand_dims(tf.cholesky(V_var) ,2)

            Fstar_mean, Fstar_var = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.F, self.F_sqrt, 1)
            Fstar_w_mean, Fstar_w_var = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, V, V_sqrt, 1)


        mean_difference = tf.Session().run(Fstar_w_mean - Fstar_mean, feed_dict=self.feed_dict)
        var_difference = tf.Session().run(Fstar_w_var - Fstar_var, feed_dict=self.feed_dict)

        self.failUnless(np.all(np.abs(mean_difference) < 1e-2))
        print 'var_diff', var_difference
        self.failUnless(np.all(np.abs(var_difference) < 1e-2))

       

     

    


if __name__ == "__main__":
    unittest.main()

