import GPflow
import theano
import numpy as np
import unittest
from theano import tensor as tt

class DiagsTest(unittest.TestCase):
    """
    The conditionals can take cholesky matrices or diagaonal matrices. 

    Here we make sure the behaviours overlap.
    """
    def setUp(self):
        self.k = GPflow.kernels.Matern32(1) + GPflow.kernels.White(1)
        self.k.k2.variance = 0.01
        self.X = tf.placeholder('float64')
        self.mu = tf.placeholder('float64')
        self.Xs = tf.placeholder('float64')
        self.sqrt = tf.placeholder('float64')

        #make theano array shenanigans
        self.free_x = tf.placeholder('float64')
        self.k.make_tf_array(self.free_x)

        self.free_x_data = self.k.get_free_state()
        # NB. with too many random data, numerics suffer
        self.rng = np.random.RandomState(0)
        self.X_data = self.rng.randn(3,1)
        self.mu_data = self.rng.randn(3,2)
        self.sqrt_data = self.rng.randn(3,2)
        self.Xs_data = self.rng.randn(50,1)

        self.chol = theano.scan(lambda x : tf.diag(x), self.sqrt.T)[0].swapaxes(0,2)

    def test_whiten(self):
        with self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, self.mu, self.sqrt)
            Fstar_mean_2, Fstar_var_2 = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, self.mu, self.chol)
            
        mean_diff = theano.function([self.X, self.Xs, self.mu, self.sqrt, self.free_x],
                                Fstar_mean_1 - Fstar_mean_2, on_unused_input='ignore')(
                                        self.X_data, self.Xs_data, self.mu_data, self.sqrt_data, self.free_x_data)

        var_diff = theano.function([self.X, self.Xs, self.mu, self.sqrt, self.free_x],
                                Fstar_var_1 - Fstar_var_2, on_unused_input='ignore')(
                                        self.X_data, self.Xs_data, self.mu_data, self.sqrt_data, self.free_x_data)

        self.failUnless(np.allclose(mean_diff, 0))
        self.failUnless(np.allclose(var_diff, 0))


    def test_nonwhiten(self):
        with self.k.tf_mode():
            Fstar_mean_1, Fstar_var_1 = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.mu, self.sqrt)
            Fstar_mean_2, Fstar_var_2 = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.mu, self.chol)
            
        mean_diff = theano.function([self.X, self.Xs, self.mu, self.sqrt, self.free_x],
                                Fstar_mean_1 - Fstar_mean_2, on_unused_input='ignore')(
                                        self.X_data, self.Xs_data, self.mu_data, self.sqrt_data, self.free_x_data)

        var_diff = theano.function([self.X, self.Xs, self.mu, self.sqrt, self.free_x],
                                Fstar_var_1 - Fstar_var_2, on_unused_input='ignore')(
                                        self.X_data, self.Xs_data, self.mu_data, self.sqrt_data, self.free_x_data)

        self.failUnless(np.allclose(mean_diff, 0))
        self.failUnless(np.allclose(var_diff, 0))










class WhitenTest(unittest.TestCase):
    def setUp(self):
        self.k = GPflow.kernels.Matern32(1) + GPflow.kernels.White(1)
        self.k.k2.variance = 0.01
        self.X = tf.placeholder('float64')
        self.F = tf.placeholder('float64')
        self.Xs = tf.placeholder('float64')

        #make theano array shenanigans
        self.free_x = tf.placeholder('float64')
        self.k.make_tf_array(self.free_x)

        self.free_x_data = self.k.get_free_state()
        # NB. with too many random data, numerics suffer
        self.rng = np.random.RandomState(0)
        self.X_data = self.rng.randn(3,1)
        self.F_data = self.rng.randn(3,1)
        self.Xs_data = self.rng.randn(50,1)

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one. 
        """
        
        with self.k.tf_mode():
            K = self.k.K(self.X)
            L = GPflow.slinalg.cholesky(K)
            V = GPflow.slinalg.Solve('lower_triangular')(L, self.F)
            Fstar_mean, Fstar_var = GPflow.conditionals.gp_predict(self.Xs, self.X, self.k, self.F)
            Fstar_w_mean, Fstar_w_var = GPflow.conditionals.gp_predict_whitened(self.Xs, self.X, self.k, V)


        mean_difference = theano.function([self.free_x, self.X, self.F, self.Xs], Fstar_w_mean - Fstar_mean)(self.free_x_data, self.X_data, self.F_data, self.Xs_data)
        var_difference = theano.function([self.free_x, self.X, self.F, self.Xs], Fstar_w_var - Fstar_var)(self.free_x_data, self.X_data, self.F_data, self.Xs_data)
                
        self.failUnless(np.all(np.abs(mean_difference) < 1e-2))
        self.failUnless(np.all(np.abs(var_difference) < 1e-2))


class WhitenTestGaussian(WhitenTest):
    def setUp(self):
        WhitenTest.setUp(self)
        self.F_sqrt = tf.placeholder('float64')
        self.F_sqrt_data = self.rng.randn(3,1)

    def test_whiten(self):
        """
        make sure that predicting using the whitened representation is the
        sameas the non-whitened one. 
        """
        with self.k.tf_mode():
            K = self.k.K(self.X)
            L = GPflow.slinalg.cholesky(K)
            V = GPflow.slinalg.Solve('lower_triangular')(L, self.F)
            Li = GPflow.slinalg.Solve('lower_triangular')(L, eye(3))
            V_var = (Li.dot(tf.diag(self.F_sqrt.flatten()**2)).dot(Li.T))
            V_sqrt = GPflow.slinalg.cholesky(V_var)[:,:,None]

            Fstar_mean, Fstar_var = GPflow.conditionals.gaussian_gp_predict(self.Xs, self.X, self.k, self.F, self.F_sqrt)
            Fstar_w_mean, Fstar_w_var = GPflow.conditionals.gaussian_gp_predict_whitened(self.Xs, self.X, self.k, V, V_sqrt)


        mean_difference = theano.function([
                            self.free_x, self.X, self.F, self.Xs],
                            Fstar_w_mean - Fstar_mean)(self.free_x_data, self.X_data, self.F_data, self.Xs_data)
        var_difference = theano.function([
                            self.free_x, self.X, self.Xs, self.F_sqrt],
                            Fstar_w_var - Fstar_var)(self.free_x_data, self.X_data, self.Xs_data, self.F_sqrt_data)

        self.failUnless(np.all(np.abs(mean_difference) < 1e-2))
        self.failUnless(np.all(np.abs(var_difference) < 1e-2))

       

     

    


if __name__ == "__main__":
    unittest.main()

