import GPflow
import numpy as np
import unittest
import tensorflow as tf


class SampleGaussianTest(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.f = lambda x: (0.5*np.sum(np.square(x)), x)
        self.x0 = np.zeros(3)

    def test_mean_cov(self):
        samples = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                                        x0=self.x0, verbose=False, thin=10, burn=0)
        mean = samples.mean(0)
        cov = np.cov(samples.T)
        self.failUnless(np.allclose(mean, np.zeros(3), 1e-1, 1e-1))
        self.failUnless(np.allclose(cov, np.eye(3), 1e-1, 1e-1))

    def test_rng(self):
        """
        Make sure all randomness can be atributed to the rng
        """
        samples1 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                                         x0=self.x0, verbose=False, thin=10, burn=0,
                                         RNG=np.random.RandomState(10))

        samples2 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                                         x0=self.x0, verbose=False, thin=10, burn=0,
                                         RNG=np.random.RandomState(10))

        samples3 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                                         x0=self.x0, verbose=False, thin=10, burn=0,
                                         RNG=np.random.RandomState(11))

        self.failUnless(np.all(samples1 == samples2))
        self.failIf(np.all(samples1 == samples3))

    def test_burn(self):
        samples = GPflow.hmc.sample_HMC(self.f, num_samples=100, Lmax=20, epsilon=0.05,
                                        x0=self.x0, verbose=False, thin=1, burn=10,
                                        RNG=np.random.RandomState(11))

        self.failUnless(samples.shape == (100, 3))
        self.failIf(np.all(samples[0] == self.x0))


class SampleModelTest(unittest.TestCase):
    """
    Create a very simple model and make sure samples form is make sense.
    """
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)

        class Quadratic(GPflow.model.Model):
            def __init__(self):
                GPflow.model.Model.__init__(self)
                self.x = GPflow.param.Param(rng.randn(2))

            def build_likelihood(self):
                return -tf.reduce_sum(tf.square(self.x))
        self.m = Quadratic()

    def test_mean(self):
        samples = self.m.sample(num_samples=200, Lmax=20, epsilon=0.05)

        self.failUnless(samples.shape == (200, 2))
        self.failUnless(np.allclose(samples.mean(0), np.zeros(2), 1e-1, 1e-1))


class SamplesDictTest(unittest.TestCase):
    def setUp(self):
        X, Y = np.random.randn(2, 10, 1)
        self.m = GPflow.gpmc.GPMC(X, Y, kern=GPflow.kernels.Matern32(1), likelihood=GPflow.likelihoods.StudentT())

    def test_samples_df(self):
        samples = self.m.sample(num_samples=20, Lmax=10, epsilon=0.05)
        sample_df = self.m.get_samples_df(samples)
        for name, trace in sample_df.iteritems():
            self.assertTrue(trace.shape[0] == 20)
            self.assertTrue(trace.iloc[0].shape == self.m.get_parameter_dict()[name].shape)
            self.assertTrue(trace.iloc[10].shape == self.m.get_parameter_dict()[name].shape)

    def test_with_fixed(self):
        self.m.kern.lengthscales.fixed = True
        samples = self.m.sample(num_samples=20, Lmax=10, epsilon=0.05)
        sample_dict = self.m.get_samples_df(samples)

        ls_trace = sample_dict['model.kern.lengthscales']
        assert np.all([np.all(v == ls_trace[0]) for v in ls_trace])


if __name__ == "__main__":
    unittest.main()
