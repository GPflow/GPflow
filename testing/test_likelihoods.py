import GPflow
import numpy as np
import unittest
from theano import tensor as tt
import theano

class TestPredictConditional(unittest.TestCase):
    """
    Here we make sure that the conditional_mean and contitional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    def setUp(self):
        self.liks = [L() for L in GPflow.likelihoods.Likelihood.__subclasses__()]
        self.x = tf.placeholder('float64')
        for l in self.liks:
            l.make_tf_array(self.x)

        self.F = np.random.randn(10,1)

    def test_mean(self):
        for l in self.liks:
            if isinstance(l, GPflow.likelihoods.Gaussian):
                #there's nothing to compile for a Gaussian.
                self.failUnless(np.allclose(l.conditional_mean(self.F), self.F))
                with l.tf_mode():
                    self.failUnless(np.allclose(l.predict_mean_and_var(self.F, self.F * 0)[0], l.conditional_mean(self.F)))
                continue
            with l.tf_mode():
                mu1 = theano.function([self.x], l.conditional_mean(self.F), on_unused_input='ignore')(l.get_free_state())
                mu2, _ = theano.function([self.x], l.predict_mean_and_var(self.F, self.F * 0), on_unused_input='ignore')(l.get_free_state())
            self.failUnless(np.allclose(mu1, mu2, 1e-6, 1e-6))


    def test_variance(self):
        for l in self.liks:
            with l.tf_mode():
                v1 = theano.function([self.x], l.conditional_variance(self.F), on_unused_input='ignore')(l.get_free_state())
                v2 = theano.function([self.x], l.predict_mean_and_var(self.F, self.F * 0)[1], on_unused_input='ignore')(l.get_free_state())
            self.failUnless(np.allclose(v1, v2, 1e-6, 1e-6))

class TestQuadrature(unittest.TestCase):
    """
    Where quadratre methods have been overwritten, make sure the new code
     does something close to the quadrature
    """
    def setUp(self):

        self.rng = np.random.RandomState()
        self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 2)
        self.Fvar = 0.01 * self.Fvar **2

    def test_var_exp(self):
        #get all the likelihoods where variational expectations has been overwritten
        liks = [l() for l in GPflow.likelihoods.Likelihood.__subclasses__()
                if l.variational_expectations.__func__ is not GPflow.likelihoods.Likelihood.variational_expectations.__func__]
        for l in liks:
            x_data = l.get_free_state()
            #make parameters if needed
            if len(x_data):
                x = tf.placeholder('float64')
                l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.variational_expectations(self.Fmu, self.Fvar, self.Y)
                F2 = GPflow.likelihoods.Likelihood.variational_expectations(l, self.Fmu, self.Fvar, self.Y)
            #compile and run the functions:
            if len(x_data):    
                F1 = theano.function([x], F1)(x_data)
                F2 = theano.function([x], F2)(x_data)
            else:
                F1 = theano.function([], F1)()
                F2 = theano.function([], F2)()
            self.failUnless(np.allclose(F1, F2, 1e-6, 1e-6))

    def test_pred_density(self):
        #get all the likelihoods where predict_density  has been overwritten
        liks = [l() for l in GPflow.likelihoods.Likelihood.__subclasses__()
                if l.predict_density.__func__ is not GPflow.likelihoods.Likelihood.predict_density.__func__]
        for l in liks:
            x_data = l.get_free_state()
            #make parameters if needed
            if len(x_data):
                x = tf.placeholder('float64')
                l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.predict_density(self.Fmu, self.Fvar, self.Y)
                F2 = GPflow.likelihoods.Likelihood.predict_density(l, self.Fmu, self.Fvar, self.Y)
            #compile and run the functions:
            if len(x_data):    
                F1 = theano.function([x], F1)(x_data)
                F2 = theano.function([x], F2)(x_data)
            else:
                F1 = theano.function([], F1)()
                F2 = theano.function([], F2)()
            self.failUnless(np.allclose(F1, F2, 1e-6, 1e-6))






        


        
    


if __name__ == "__main__":
    unittest.main()

