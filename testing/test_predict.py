import GPflow
import numpy as np
import unittest
import tensorflow as tf


class TestGaussian(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100,2)
        self.Y = self.rng.randn(100, 1)
        self.kern = GPflow.kernels.Matern32(2) + GPflow.kernels.White(1)
        self.Xtest = self.rng.randn(10, 2)
        self.Ytest = self.rng.randn(10, 1)

        # make a Gaussian model
        self.m = GPflow.gpr.GPR(self.X, self.Y, kern=self.kern)

    def test_all(self):
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
    """ 
    this base class requires inherriting to specify the model.

    This test structure is more complex that, say, looping over the models, but
    makses all the tests much smaller and so less prone to erroring out. Also,
    if a test fails, it should be clearer where the error is. 
    """
    def setUp(self):
        tf.reset_default_graph()
        self.input_dim = 3
        self.output_dim = 2
        self.N = 20
        self.Ntest = 30
        self.M = 5
        rng = np.random.RandomState(0)
        self.num_samples = 5
        self.samples_shape = (self.num_samples, self.Ntest, self.output_dim)
        self.covar_shape = (self.Ntest, self.Ntest, self.output_dim)
        self.X, self.Y, self.Z, self.Xtest = rng.randn(self.N, self.input_dim),\
                              rng.randn(self.N, self.output_dim),\
                              rng.randn(self.M, self.input_dim),\
                              rng.randn(self.Ntest, self.input_dim)
        self.k = lambda: GPflow.kernels.Matern32(self.input_dim)
        self.model = GPflow.gpr.GPR(self.X, self.Y, kern=self.k())

    def test_cov(self):
        mu1, var = self.model.predict_f(self.Xtest)
        mu2, covar = self.model.predict_f_full_cov(self.Xtest)
        self.failUnless(np.all(mu1 == mu2))
        self.failUnless(covar.shape == self.covar_shape)
        self.failUnless(var.shape == (self.Ntest, self.output_dim))
        for i in range(self.output_dim):
            self.failUnless(np.allclose(var[:, i], np.diag(covar[:, :, i])))

    def test_samples(self):
        samples = self.model.predict_f_samples(self.Xtest, self.num_samples)
        self.failUnless(samples.shape == self.samples_shape)


class TestFullCovSGPR(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.sgpr.SGPR(self.X, self.Y, Z=self.Z, kern=self.k())


class TestFullCovGPRFITC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.sgpr.GPRFITC(self.X, self.Y,
                                         Z=self.Z, kern=self.k())


class TestFullCovSVGP1(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.svgp.SVGP(self.X, self.Y, Z=self.Z, kern=self.k(),
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      whiten=False, q_diag=True)


class TestFullCovSVGP2(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.svgp.SVGP(self.X, self.Y, Z=self.Z, kern=self.k(),
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      whiten=True, q_diag=False)


class TestFullCovSVGP3(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.svgp.SVGP(self.X, self.Y, Z=self.Z, kern=self.k(),
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      whiten=True, q_diag=True)


class TestFullCovSVGP4(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.svgp.SVGP(self.X, self.Y, Z=self.Z, kern=self.k(),
                                      likelihood=GPflow.likelihoods.Gaussian(),
                                      whiten=True, q_diag=False)


class TestFullCovVGP(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.vgp.VGP(self.X, self.Y, kern=self.k(),
                                    likelihood=GPflow.likelihoods.Gaussian())


class TestFullCovGPMC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.gpmc.GPMC(self.X, self.Y, kern=self.k(),
                                      likelihood=GPflow.likelihoods.Gaussian())


class TestFullCovSGPMC(TestFullCov):
    def setUp(self):
        TestFullCov.setUp(self)
        self.model = GPflow.sgpmc.SGPMC(self.X, self.Y, kern=self.k(),
                                        likelihood=GPflow.likelihoods.Gaussian(),
                                        Z=self.Z)


if __name__ == "__main__":
    unittest.main()
