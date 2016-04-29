import GPflow

import tensorflow as tf
import numpy as np
import unittest

class TestMeanFuncs(unittest.TestCase):
    """
    Test the output shape for basic and compositional mean functions, also
    check that the combination of mean functions returns the correct clas
    """
    def setUp(self):
        self.input_dim=3
        self.output_dim=2
        self.N=20
        rng = np.random.RandomState(0)
        self.mfs = [GPflow.mean_functions.Zero(),
                    GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim), rng.randn(self.output_dim)),
                    GPflow.mean_functions.Constant(rng.randn(self.output_dim))]
                    
        
        self.composition_mfs_add = []
        self.composition_mfs_mult = []
        
        for mean_f1 in self.mfs:
            self.composition_mfs_add.extend([mean_f1 + mean_f2 for mean_f2 in self.mfs])
            self.composition_mfs_mult.extend([mean_f1 * mean_f2 for mean_f2 in self.mfs])
        
        self.composition_mfs = self.composition_mfs_add + self.composition_mfs_mult
        self.x = tf.placeholder('float64')

        for mf in self.mfs:
            mf.make_tf_array(self.x)
        for compmf in self.composition_mfs:
            compmf.make_tf_array(self.x)
            
        
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

    def test_combination_types(self):
        self.failUnless(all(isinstance(mfAdd, GPflow.mean_functions.Additive) for mfAdd in self.composition_mfs_add))
        self.failUnless(all(isinstance(mfMult, GPflow.mean_functions.Product) for mfMult in self.composition_mfs_mult))
        
        
class TestModelCompositionOperations(unittest.TestCase):
    """
    Tests that operator precedence is correct and zero unary operations, i.e.
    adding 0, multiplying by 1, adding x and then subtracting etc. do not change
    the mean function
    """
    def setUp(self):
        self.input_dim=3
        self.output_dim=2
        self.N=20
        rng = np.random.RandomState(0)
        
        X = rng.randn(self.N, self.input_dim)
        Y = rng.randn(self.N, self.output_dim)
        self.Xtest =  rng.randn(30, self.output_dim)
        
        zero = GPflow.mean_functions.Zero()
        one = GPflow.mean_functions.Constant(np.ones(self.output_dim))
        
        linear1 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim), rng.randn(self.output_dim)),
        linear2 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim), rng.randn(self.output_dim)),
        linear3 = GPflow.mean_functions.Linear(rng.randn(self.input_dim, self.output_dim), rng.randn(self.output_dim)),
        
        const1 = GPflow.mean_functions.Constant(rng.randn(self.output_dim))
        const2 = GPflow.mean_functions.Constant(rng.randn(self.output_dim))
        const3 = GPflow.mean_functions.Constant(rng.randn(self.output_dim))
        
        
        const1inv = GPflow.mean_functions.Constant(-1 * const1.c)
        linear1inv = GPflow.mean_functions.Linear(-1 * linear1.A, linear1.b * -1)
        
        #a * (b + c)
        const_set1 = GPflow.mean_functions.Product(const1,
                                                  GPflow.mean_functions.Additive(const2, const3))
        linear_set1 = GPflow.mean_functions.Product(linear1,
                                                  GPflow.mean_functions.Additive(linear2, linear3))

        #ab + ac
        const_set2 = GPflow.mean_functions.Additive(GPflow.mean_functions.Product(const1, const2),
                                                   GPflow.mean_functions.Product(const1, const3))

        linear_set2 = GPflow.mean_functions.Additive(GPflow.mean_functions.Product(linear1, linear2),
                                                     GPflow.mean_functions.Product(linear1, linear3))
        #a-a = 0, (a + b) -a = b = a + (b - a)
        
        linear1_minus_linear1 =  GPflow.mean_functions.Additive(linear1, linear1inv)   
        const1_minus_const1=  GPflow.mean_functions.Additive(const1, const1inv)

        comp_minus_constituent1 = GPflow.mean_functions.Additive(GPflow.mean_functions.Additive(linear1, linear2),
                                                                      linear1inv)
        comp_minus_constituent2 = GPflow.mean_functions.Additive(linear1,
                                                                      GPflow.mean_functions.Additive(linear2,
                                                                                                     linear1inv))  
                                                    
        k = GPflow.kernels.Bias(self.input_dim)        
        
        self.m_linear_set1 = GPflow.gpr.GPR(X, Y, mean_function=linear_set1, kern=k)
        self.m_linear_set2 = GPflow.gpr.GPR(X, Y, mean_function=linear_set2, kern=k)
        
        self.m_const_set1 = GPflow.gpr.GPR(X, Y, mean_function=const_set1, kern=k)
        self.m_const_set2 = GPflow.gpr.GPR(X, Y, mean_function=const_set2, kern=k)
        
        self.m_linear_min_linear= GPflow.gpr.GPR(X, Y, mean_function=linear1_minus_linear1, kern=k)
        self.m_const_min_const = GPflow.gpr.GPR(X, Y, mean_function=const1_minus_const1, kern=k)        
        
        self.m_constituent = GPflow.gpr.GPR(X, Y, mean_function=linear2, kern=k) 
        self.m_zero = GPflow.gpr.GPR(X, Y, mean_function=zero, kern=k)        
        
        self.m_comp_minus_constituent1 = GPflow.gpr.GPR(X, Y, mean_function=comp_minus_constituent1, kern=k)
        self.m_comp_minus_constituent2 = GPflow.gpr.GPR(X, Y, mean_function=comp_minus_constituent2, kern=k)

             
    def test_precedence(self):
        mu1_lin, v1_lin = self.m_linear_set1.predict_f(self.Xtest)
        mu2_lin, v2_lin = self.m_linear_set2.predict_f(self.Xtest)
        
        mu1_const, v1_const = self.m_const_set1.predict_f(self.Xtest)
        mu2_const, v2_const = self.m_const_set2.predict_f(self.Xtest)
        
        self.failUnless(np.all(v1_lin==v1_lin))
        self.failUnless(np.all(mu1_lin==mu2_lin))
        
        self.failUnless(np.all(v1_const==v2_const))
        self.failUnless(np.all(mu1_const==mu2_const))
            
    def test_inverse_operations(self):
        mu1_lin_min_lin, v1_lin_min_lin = self.m_linear_min_linear.predict_f(self.Xtest)
        mu1_const_min_const, v1_const_min_const= self.m_const_min_const.predict_f(self.Xtest)        
        
        mu1_comp_min_constituent1, v1_comp_min_constituent1 = self.m_comp_minus_constituent1.predict_f(self.Xtest)
        mu1_comp_min_constituent2, v1_comp_min_constituent2 = self.m_comp_minus_constituent2.predict_f(self.Xtest)
        
        mu_constituent, v_constituent = self.m_constituent.predict_f(self.Xtest)
        mu_zero, v_zero= self.m_zero.predict_f(self.Xtest)
        
        self.failUnless(np.all(mu1_lin_min_lin, mu_zero))
        self.failUnless(np.all(mu1_const_min_const, mu_zero))
        
        self.failUnless(np.all(np.isclose(mu1_comp_min_constituent1, mu_constituent)))
        self.failUnless(np.all(np.isclose(mu1_comp_min_constituent2, mu_constituent)))
        self.failUnless(np.all(np.isclose(mu1_comp_min_constituent1, mu1_comp_min_constituent2)))           


class TestModelsWithMeanFuncs(unittest.TestCase):
    """
    Simply check that all models have a higher prediction with a constant mean
    function than with a zero mean function.
    
    For compositions of mean functions check that multiplication/ addition of 
    a constant results in a higher prediction, whereas addition of zero/
    mutliplication with one does not.
    
    """

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
        one = GPflow.mean_functions.Constant(np.ones(self.output_dim))
        
        
        mult = GPflow.mean_functions.Product(const,const)
        add = GPflow.mean_functions.Additive(const,const)
        
        addzero = GPflow.mean_functions.Additive(const, zero)
        multone = GPflow.mean_functions.Product(const, one)

        self.models_with, self.models_without, self.models_mult, self.models_add, \
        self.models_addzero, self.models_multone =\
                [[GPflow.gpr.GPR(X, Y, mean_function=mf, kern=k()),
                  GPflow.sgpr.SGPR(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.sgpr.GPRFITC(X, Y, mean_function=mf, Z=Z, kern=k()),
                  GPflow.svgp.SVGP(X, Y, mean_function=mf, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.vgp.VGP(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.gpmc.GPMC(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.sgpmc.SGPMC(X, Y, mean_function=mf, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), Z=Z)] for mf in (const, zero, mult, add, addzero, multone)]

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
    
    def test_addZero_meanfuction(self):
        #TODO: this does not work yet? fix precision?
        for m_add, m_addZero in zip(self.models_add, self.models_addzero):
            mu1, v1 = m_add.predict_f(self.Xtest)
            mu2, v2 = m_addZero.predict_f(self.Xtest)
            self.failUnless(np.all(np.isclose(mu1,mu2)))
            self.failUnless(np.all(np.isclose(v1, v2)))
    def test_multOne_meanfuction(self):
        #TODO: this does not work yet?
        for m_mult, m_multOne in zip(self.models_mult, self.models_multone):
            mu1, v1 = m_mult.predict_f(self.Xtest)
            mu2, v2 = m_multOne.predict_f(self.Xtest)
            self.failUnless(np.all(np.isclose(mu1,mu2)))
            self.failUnless(np.all(np.isclose(v1, v2)))

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestModelCompositionOperations)
    unittest.TextTestRunner(verbosity=2).run(suite)
    #unittest.main()

