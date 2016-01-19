import GPflow
import tensorflow as tf
import numpy as np
import unittest

class TestGaussian(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100,2)
        self.Y = self.rng.randn(100,1)
        self.kern = GPflow.kernels.Matern32(2) + GPflow.kernels.White(1)
        self.Xtest = self.rng.randn(10,2)
        self.Ytest = self.rng.randn(10,1)

        #make a Gaussian model
        self.m = GPflow.gpr.GPR(self.X, self.Y, kern=self.kern)
        self.m._compile()


    def test_mean_variance(self):
        mu_f, var_f = self.m.predict_f(self.Xtest)
        mu_y, var_y = self.m.predict_y(self.Xtest)

        self.failUnless(np.allclose(mu_f, mu_y))
        self.failUnless(np.allclose(var_f, var_y - 1.))

    def test_density(self):
        mu_y, var_y = self.m.predict_y(self.Xtest)
        density = self.m.predict_density(self.Xtest, self.Ytest)

        density_hand = -0.5*np.log(2*np.pi) - 0.5*np.log(var_y) - 0.5*np.square(mu_y - self.Ytest)/var_y
        self.failUnless(np.allclose(density_hand, density))

    def test_recompile(self):
        mu_f, var_f = self.m.predict_f(self.Xtest)
        mu_y, var_y = self.m.predict_y(self.Xtest)
        density = self.m.predict_density(self.Xtest, self.Ytest)

        #change a fix and see if these things still compile
        self.m.likelihood.variance = 0.2
        self.m.likelihood.variance.fixed = True

        #this will fail unless a recompile has been triggered
        mu_f, var_f = self.m.predict_f(self.Xtest)
        mu_y, var_y = self.m.predict_y(self.Xtest)
        density = self.m.predict_density(self.Xtest, self.Ytest)



        














if __name__ == "__main__":
    unittest.main()

