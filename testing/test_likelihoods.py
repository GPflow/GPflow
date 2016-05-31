import GPflow
import tensorflow as tf
import numpy as np
import unittest


class TestPredictConditional(unittest.TestCase):
    """
    Here we make sure that the conditional_mean and contitional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.liks = [L() for L in GPflow.likelihoods.Likelihood.__subclasses__()]

        #some likelihoods are additionally tested with non-standard links
        self.liks.append(GPflow.likelihoods.Poisson(invlink=tf.square))
        self.liks.append(GPflow.likelihoods.Exponential(invlink=tf.square))
        self.liks.append(GPflow.likelihoods.Gamma(invlink=tf.square))
        sigmoid = lambda x : 1./(1 + tf.exp(-x))
        self.liks.append(GPflow.likelihoods.Bernoulli(invlink=sigmoid))

        self.x = tf.placeholder('float64')
        for l in self.liks:
            l.make_tf_array(self.x)

        self.F = tf.placeholder(tf.float64)
        rng = np.random.RandomState(0)
        self.F_data = rng.randn(10,2)
        self.Y = rng.rand(10,2)

    def test_mean(self):
        for l in self.liks:
            with l.tf_mode():
                mu1 = tf.Session().run(l.conditional_mean(self.F), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                mu2, _ = tf.Session().run(l.predict_mean_and_var(self.F, self.F * 0), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
            self.failUnless(np.allclose(mu1, mu2, 1e-6, 1e-6))


    def test_variance(self):
        for l in self.liks:
            with l.tf_mode():
                v1 = tf.Session().run(l.conditional_variance(self.F), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                v2 = tf.Session().run(l.predict_mean_and_var(self.F, self.F * 0)[1], feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
            self.failUnless(np.allclose(v1, v2, 1e-6, 1e-6))

    def test_var_exp(self):
        """
        Here we make sure that the variational_expectations gives the same result
        as logp if the latent function has no uncertainty.
        """
        for l in self.liks:
            with l.tf_mode():
                r1 = tf.Session().run(l.logp(self.F, self.Y), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                r2 = tf.Session().run(l.variational_expectations(self.F, self.F * 0, self.Y), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
            self.failUnless(np.allclose(r1, r2, 1e-6, 1e-6))



class TestQuadrature(unittest.TestCase):
    """
    Where quadratre methods have been overwritten, make sure the new code
     does something close to the quadrature
    """
    def setUp(self):
        tf.reset_default_graph()

        self.rng = np.random.RandomState()
        self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 2)
        self.Y = self.Y**2
        self.Fvar = 0.01 * self.Fvar **2

    def test_var_exp(self):
        #get all the likelihoods where variational expectations has been overwritten
        liks = [l() for l in GPflow.likelihoods.Likelihood.__subclasses__()
                if l.variational_expectations.__func__ is not GPflow.likelihoods.Likelihood.variational_expectations.__func__]
        for l in liks:
            x_data = l.get_free_state()
            x = tf.placeholder('float64')
            l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.variational_expectations(self.Fmu, self.Fvar, self.Y)
                F2 = GPflow.likelihoods.Likelihood.variational_expectations(l, self.Fmu, self.Fvar, self.Y)
            #compile and run the functions:
            F1 = tf.Session().run(F1, feed_dict={x: x_data})
            F2 = tf.Session().run(F2, feed_dict={x: x_data})
            self.failUnless(np.allclose(F1, F2, 1e-6, 1e-6))

    def test_pred_density(self):
        #get all the likelihoods where predict_density  has been overwritten
        liks = [l() for l in GPflow.likelihoods.Likelihood.__subclasses__()
                if l.predict_density.__func__ is not GPflow.likelihoods.Likelihood.predict_density.__func__]
        for l in liks:
            x_data = l.get_free_state()
            #make parameters if needed
            x = tf.placeholder('float64')
            l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.predict_density(self.Fmu, self.Fvar, self.Y)
                F2 = GPflow.likelihoods.Likelihood.predict_density(l, self.Fmu, self.Fvar, self.Y)
            #compile and run the functions:
            F1 = tf.Session().run(F1, feed_dict={x: x_data})
            F2 = tf.Session().run(F2, feed_dict={x: x_data})
            self.failUnless(np.allclose(F1, F2, 1e-6, 1e-6))


if __name__ == "__main__":
    unittest.main()

