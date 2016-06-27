import GPflow
import tensorflow as tf
import numpy as np
import unittest

class TestSetup(object):
    def __init__( self, likelihood, Y, tolerance ):
        self.likelihood, self.Y, self.tolerance = likelihood, Y, tolerance
        self.is_analytic = likelihood.predict_density.__func__ is not GPflow.likelihoods.Likelihood.predict_density.__func__
        
def getTestSetups(includeMultiClass=True,addNonStandardLinks=False):
    test_setups = []
    rng = np.random.RandomState(1)
    for likelihoodClass in GPflow.likelihoods.Likelihood.__subclasses__():
        if likelihoodClass!=GPflow.likelihoods.MultiClass:
            test_setups.append( TestSetup( likelihoodClass() , rng.rand(10,2) , 1e-6 ) )
        elif includeMultiClass:
            sample = rng.randn(10,2)
            #Multiclass needs a less tight tolerance due to presence of clipping.
            tolerance = 1e-3
            test_setups.append( TestSetup( likelihoodClass(2) ,  np.argmax(sample, 1).reshape(-1,1) , tolerance )  )
    
    if addNonStandardLinks:        
        test_setups.append( TestSetup( GPflow.likelihoods.Poisson(invlink=tf.square) , rng.rand(10,2) , 1e-6 ) )
        test_setups.append( TestSetup( GPflow.likelihoods.Exponential(invlink=tf.square) , rng.rand(10,2) , 1e-6 ) )
        test_setups.append( TestSetup( GPflow.likelihoods.Gamma(invlink=tf.square) , rng.rand(10,2) , 1e-6 ) )
        sigmoid = lambda x : 1./(1 + tf.exp(-x))
        test_setups.append( TestSetup( GPflow.likelihoods.Bernoulli(invlink=sigmoid) , rng.rand(10,2) , 1e-6 ) )
    return test_setups

class TestPredictConditional(unittest.TestCase):
    """
    Here we make sure that the conditional_mean and contitional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.test_setups = getTestSetups(addNonStandardLinks=True)
        
        self.x = tf.placeholder('float64')
        for test_setup in self.test_setups:
            test_setup.likelihood.make_tf_array(self.x)

        self.F = tf.placeholder(tf.float64)
        rng = np.random.RandomState(0)
        self.F_data = rng.randn(10,2)

    def test_mean(self):
        for test_setup in self.test_setups:
            l = test_setup.likelihood
            with l.tf_mode():
                mu1 = tf.Session().run(l.conditional_mean(self.F), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                mu2, _ = tf.Session().run(l.predict_mean_and_var(self.F, self.F * 0), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
            self.failUnless(np.allclose(mu1, mu2, test_setup.tolerance, test_setup.tolerance))

    def test_variance(self):
        for test_setup in self.test_setups:
            l = test_setup.likelihood
            with l.tf_mode():
                v1 = tf.Session().run(l.conditional_variance(self.F), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                v2 = tf.Session().run(l.predict_mean_and_var(self.F, self.F * 0)[1], feed_dict={self.x: l.get_free_state(), self.F:self.F_data})   
            self.failUnless(np.allclose(v1, v2, test_setup.tolerance, test_setup.tolerance))

    def test_var_exp(self):
        """
        Here we make sure that the variational_expectations gives the same result
        as logp if the latent function has no uncertainty.
        """
        for test_setup in self.test_setups:
            l = test_setup.likelihood
            y = test_setup.Y
            with l.tf_mode():
                r1 = tf.Session().run(l.logp(self.F, y), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})
                r2 = tf.Session().run(l.variational_expectations(self.F, self.F * 0,test_setup.Y), feed_dict={self.x: l.get_free_state(), self.F:self.F_data})   
            self.failUnless(np.allclose(r1, r2, test_setup.tolerance, test_setup.tolerance))

class TestQuadrature(unittest.TestCase):
    """
    Where quadratre methods have been overwritten, make sure the new code
     does something close to the quadrature
    """
    def setUp(self):
        tf.reset_default_graph()

        self.rng = np.random.RandomState()
        self.Fmu, self.Fvar, self.Y = self.rng.randn(3, 10, 2)
        self.Fvar = 0.01 * self.Fvar **2
        self.test_setups = getTestSetups(includeMultiClass=False)

    def test_var_exp(self):
        #get all the likelihoods where variational expectations has been overwritten
                
        for test_setup in self.test_setups:
            if not test_setup.is_analytic:
                continue
            l = test_setup.likelihood
            y = test_setup.Y
            x_data = l.get_free_state()
            x = tf.placeholder('float64')
            l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.variational_expectations(self.Fmu, self.Fvar, y)
                F2 = GPflow.likelihoods.Likelihood.variational_expectations(l, self.Fmu, self.Fvar, y)
            #compile and run the functions:
            F1 = tf.Session().run(F1, feed_dict={x: x_data})
            F2 = tf.Session().run(F2, feed_dict={x: x_data})
            self.failUnless(np.allclose(F1, F2, test_setup.tolerance, test_setup.tolerance))

    def test_pred_density(self):
        #get all the likelihoods where predict_density  has been overwritten.
        
        for test_setup in self.test_setups:
            if not test_setup.is_analytic:
                continue
            l = test_setup.likelihood
            y = test_setup.Y
            x_data = l.get_free_state()
            #make parameters if needed
            x = tf.placeholder('float64')
            l.make_tf_array(x)
            #'build' the functions
            with l.tf_mode():
                F1 = l.predict_density(self.Fmu, self.Fvar, y)
                F2 = GPflow.likelihoods.Likelihood.predict_density(l, self.Fmu, self.Fvar, y)
            #compile and run the functions:
            F1 = tf.Session().run(F1, feed_dict={x: x_data})
            F2 = tf.Session().run(F2, feed_dict={x: x_data})
            self.failUnless(np.allclose(F1, F2, test_setup.tolerance, test_setup.tolerance))

class TestRobustMaxMulticlass(unittest.TestCase):
    """
    Some specialized tests to the multiclass likelihood with RobustMax inverse link function.
    """
    def setUp(self):
        tf.reset_default_graph()
    
    def testSymmetric(self):
        """
        This test is based on the observation that for 
        symmetric inputs the class predictions must have equal probability.
        """
        nClasses = 5
        nPoints = 10
        tolerance = 1e-4
        epsilon = 1e-3
        F = tf.placeholder(tf.float64)
        x = tf.placeholder('float64')
        F_data = np.ones( (nPoints, nClasses) )
        l = GPflow.likelihoods.MultiClass(nClasses)
        l.invlink.epsilon = epsilon
        rng = np.random.RandomState(1)
        Y = rng.randint( nClasses, size = (nPoints,1) )
        with l.tf_mode():
            mu, _ = tf.Session().run(l.predict_mean_and_var(F,F), feed_dict={x: l.get_free_state(), F:F_data})  
            pred = tf.Session().run(l.predict_density(F,F,Y), feed_dict={x: l.get_free_state(), F:F_data})  
            variational_expectations = tf.Session().run(l.variational_expectations(F,F,Y), feed_dict={x: l.get_free_state(), F:F_data}) 
        self.failUnless( np.allclose( mu , np.ones((nPoints,nClasses))/nClasses, tolerance, tolerance ) )
        self.failUnless( np.allclose( pred , np.ones((nPoints,1))/nClasses, tolerance, tolerance ) )
        validation_variational_expectation = 1./nClasses * np.log( 1.- epsilon ) + (1. - 1./nClasses ) * np.log( epsilon / (nClasses - 1) )
        self.failUnless( np.allclose( variational_expectations , np.ones((nPoints,1))*validation_variational_expectation, tolerance, tolerance ) )
            
if __name__ == "__main__":
    unittest.main()

