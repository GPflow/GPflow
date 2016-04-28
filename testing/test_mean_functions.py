import GPflow

import tensorflow as tf
import numpy as np
import unittest

class TestMeanFuncs(unittest.TestCase):
    def setUp(self):
        self.input_dim=3
        self.output_dim=2
        self.N=20
        rng = np.random.RandomState(0)
        self.mfs = [GPflow.mean_functions.Zero(),
                    GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim), rng.randn(self.output_dim)),
                    GPflow.mean_functions.Constant(rng.randn(self.output_dim))]
                    
        
        self.composition_mfs = []
        for mean_f1 in self.mfs:
            self.composition_mfs.extend([mean_f1 + mean_f2 for mean_f2 in self.mfs])
            self.composition_mfs.extend([mean_f1 * mean_f2 for mean_f2 in self.mfs])
        
        
        self.x = tf.placeholder('float64')
        for mf in self.mfs:
            mf.make_tf_array(self.x)
        

        self.X = tf.placeholder(tf.float64, [self.N, self.input_dim])
        self.X_data = np.random.randn(self.N, self.input_dim)

    def test_basic_output_shape(self):
        for mf in self.mfs:
            with mf.tf_mode():
                Y = tf.Session().run(mf(self.X), feed_dict={self.x: mf.get_free_state(), self.X:self.X_data})
            self.failUnless(Y.shape in [(self.N, self.output_dim), (self.N, 1)])
    
    def test_composition_output_shape(self):
        for comp_mf in self.composition_mfs:
            with comp_mf.tf_mode():
                
                Y = tf.Session().run(comp_mf(self.X), feed_dict={self.x: comp_mf.get_free_state(), self.X:self.X_data})
            self.failUnless(Y.shape in [(self.N, self.output_dim), (self.N, 1)])
            


class TestModelsWithMeanFuncs(unittest.TestCase):
    """
    Simply check that all models have a higher prediction with a constant mean
    function than with a zero mean function.
    
    """
    
    #TODO: Add 0, multiply with one --> not higher prediction
    def setUp(self):
        self.input_dim=3
        self.output_dim=2
        self.N=20
        self.Ntest=30
        self.M=5
        rng = np.random.RandomState(0)
        X, Y, Z, self.Xtest = rng.randn(self.N, self.input_dim),\
                              rng.randn(self.N, self.output_dim),\
                              rng.randn(self.M, self.input_dim),\
                              rng.randn(self.Ntest, self.input_dim)
        k = lambda : GPflow.kernels.Matern32(self.input_dim)
        zero = GPflow.mean_functions.Zero()
        const = GPflow.mean_functions.Constant(np.ones(self.output_dim) * 10)
        
        
        mult = GPflow.mean_functions.Product(const,const)
        add = GPflow.mean_functions.Additive(const,const)

        self.models_with, self.models_without, self.models_mult, self.models_add =\
                [[GPflow.gpr.GPR(X, Y, mean_function=mf, kern=k()),
                  GPflow.sgpr.SGPR(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.sgpr.GPRFITC(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.svgp.SVGP(X, Y, mean_function=mf, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.gpmc.GPMC(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.sgpmc.SGPMC(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), Z=Z)] for mf in (const, zero, mult, add)]

    def test_basic_mean_function(self):
        for m_with, m_without in zip(self.models_with, self.models_without):
            mu1, v1 = m_with.predict_f(self.Xtest)
            mu2, v2 = m_without.predict_f(self.Xtest)
            self.failUnless(np.all(v1==v2))
            self.failIf(np.all(mu1 == mu2))
    
    def test_add_mean_function(self):
        for m_add, m_const in zip(self.models_add, self.models_with):
            mu1, v1 = m_add.predict_f(self.Xtest)
            mu2, v2 = m_const.predict_f(self.Xtest)
            self.failUnless(np.all(v1==v2))
            self.failIf(np.all(mu1 == mu2))
    
    def test_mult_mean_function(self):
        for m_mult, m_const in zip(self.models_mult, self.models_with):
            mu1, v1 = m_mult.predict_f(self.Xtest)
            mu2, v2 = m_const.predict_f(self.Xtest)
            self.failUnless(np.all(v1==v2))
            self.failIf(np.all(mu1 == mu2))


class TestCombinationsOfMeanFunctions(unittest.TestCase):
    """
    Check that the combination of mean functions returns the correct class and
    operator precedence is correct    
    """
    def setUp(self):
        #TODO: setup a list of random additions and one of random multiplications
        print("test")
        pass
    def test_combination_types(self):
        #TODO: combine elements and test their type
        self.assertEqual('foo'.upper(), 'FOO')
        pass
    def test_precedence(self):
        #TODO: check a + b * c, a* b +c etc
        pass
    def test_zero_operations(self):
        #TODO: check a - a && a * zero for all a & combinations
        pass


if __name__ == "__main__":
      
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelsWithMeanFuncs)
    unittest.TextTestRunner(verbosity=2).run(suite)

