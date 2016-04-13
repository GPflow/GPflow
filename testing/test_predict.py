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


class TestFullCov(unittest.TestCase):
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
        self.models = [GPflow.gpr.GPR(X, Y, kern=k()),
                  GPflow.sgpr.SGPR(X, Y, Z=Z, kern=k()),
                  GPflow.sgpr.GPRFITC(X, Y, Z=Z, kern=k()),
                  GPflow.svgp.SVGP(X, Y, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), whiten=False, q_diag=True),
                  GPflow.svgp.SVGP(X, Y, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), whiten=True, q_diag=False),
                  GPflow.svgp.SVGP(X, Y, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), whiten=True, q_diag=True),
                  GPflow.svgp.SVGP(X, Y, Z=Z, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), whiten=True, q_diag=False),
                  GPflow.vgp.VGP(X, Y, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.vgp.VGP(X, Y, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.gpmc.GPMC(X, Y, kern=k(), likelihood=GPflow.likelihoods.Gaussian()),
                  GPflow.sgpmc.SGPMC(X, Y, kern=k(), likelihood=GPflow.likelihoods.Gaussian(), Z=Z)]

    def test_cov(self):
        for m in self.models:
            mu1, var = m.predict_f(self.Xtest)
            mu2, covar = m.predict_f_full_cov(self.Xtest)
            self.failUnless(np.all(mu1==mu2))
            self.failUnless(covar.shape == (self.Ntest, self.Ntest, self.output_dim))
            self.failUnless(var.shape == (self.Ntest, self.output_dim))
            for i in range(self.output_dim):
                self.failUnless(np.allclose(var[:,i] , np.diag(covar[:,:,i])))
    
    def test_samples(self):
        for m in self.models:
            samples = m.predict_f_samples(self.Xtest, 5)
            self.failUnless(samples.shape==(5, self.Xtest.shape[0], self.output_dim))


if __name__ == "__main__":
    unittest.main()

