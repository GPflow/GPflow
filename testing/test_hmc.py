import gpflow
import numpy as np
import unittest
import tensorflow as tf

try:
    import pandas
except ImportError:
    pandas = None

from nose.plugins.attrib import attr

from testing.gpflow_testcase import GPflowTestCase
from nose.plugins.attrib import attr

@attr(speed='slow')
class SampleGaussianTest(GPflowTestCase):
    def setUp(self):
        self.f = lambda x: (0.5 * np.sum(np.square(x)), x)
        self.x0 = np.zeros(3)

    def test_mean_cov(self):
        samples = gpflow.hmc.sample_HMC(
            self.f, num_samples=1000, Lmin=10, Lmax=20, epsilon=0.05,
            x0=self.x0, verbose=False, thin=10, burn=0)
        mean = samples.mean(0)
        cov = np.cov(samples.T)
        self.assertTrue(np.allclose(mean, np.zeros(3), 1e-1, 1e-1))
        self.assertTrue(np.allclose(cov, np.eye(3), 1e-1, 1e-1))

    def test_rng(self):
        """
        Make sure all randomness can be atributed to the rng
        """
        samples1 = gpflow.hmc.sample_HMC(
            self.f, num_samples=1000, Lmin=10, Lmax=20, epsilon=0.05,
            x0=self.x0, verbose=False, thin=10, burn=0,
            RNG=np.random.RandomState(10))

        samples2 = gpflow.hmc.sample_HMC(
            self.f, num_samples=1000, Lmin=10, Lmax=20, epsilon=0.05,
            x0=self.x0, verbose=False, thin=10, burn=0,
            RNG=np.random.RandomState(10))

        samples3 = gpflow.hmc.sample_HMC(
            self.f, num_samples=1000, Lmin=10, Lmax=20, epsilon=0.05,
            x0=self.x0, verbose=False, thin=10, burn=0,
            RNG=np.random.RandomState(11))

        self.assertTrue(np.all(samples1 == samples2))
        self.assertFalse(np.all(samples1 == samples3))

    def test_burn(self):
        samples = gpflow.hmc.sample_HMC(self.f, num_samples=100, Lmin=10, Lmax=20, epsilon=0.05,
                                        x0=self.x0, verbose=False, thin=1, burn=10,
                                        RNG=np.random.RandomState(11))

        self.assertTrue(samples.shape == (100, 3))
        self.assertFalse(np.all(samples[0] == self.x0))

    def test_return_logprobs(self):
        s, logps = gpflow.hmc.sample_HMC(self.f, num_samples=100, Lmin=10, Lmax=20, epsilon=0.05,
                                         x0=self.x0, verbose=False, thin=1, burn=10,
                                         RNG=np.random.RandomState(11), return_logprobs=True)




class SampleModelTest(GPflowTestCase):
    """
    Create a very simple model and make sure samples form is make sense.
    """
    def setUp(self):
        rng = np.random.RandomState(0)
        class Quadratic(gpflow.model.Model):
            def __init__(self):
                gpflow.model.Model.__init__(self)
                self.x = gpflow.param.Param(rng.randn(2))
            def build_likelihood(self):
                return -tf.reduce_sum(tf.square(self.x))
        self.m = Quadratic()

    def test_mean(self):
        with self.test_session():
            samples = self.m.sample(num_samples=400, Lmin=10, Lmax=20, epsilon=0.05)
            self.assertTrue(samples.shape == (400, 2))
            self.assertTrue(np.allclose(samples.mean(0), np.zeros(2), 1e-1, 1e-1))

    def test_return_logprobs(self):
        with self.test_session():
            s, logps = self.m.sample(num_samples=200, Lmax=20,
                                     epsilon=0.05, return_logprobs=True)


class SamplesDictTest(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            X, Y = np.random.randn(2, 10, 1)
            self.m = gpflow.gpmc.GPMC(X, Y, kern=gpflow.kernels.Matern32(1), likelihood=gpflow.likelihoods.StudentT())

    @unittest.skipIf(pandas is None, "Pandas module required for dataframes.")
    def test_samples_df(self):
        with self.test_session():
            samples = self.m.sample(num_samples=20, Lmax=10, epsilon=0.05)
            sample_df = self.m.get_samples_df(samples)
            for name, trace in sample_df.iteritems():
                self.assertTrue(trace.shape[0] == 20)
                self.assertTrue(trace.iloc[0].shape == self.m.get_parameter_dict()[name].shape)
                self.assertTrue(trace.iloc[10].shape == self.m.get_parameter_dict()[name].shape)

    @unittest.skipIf(pandas is None, "Pandas module required for dataframes.")
    def test_with_fixed(self):
        with self.test_session():
            self.m.kern.lengthscales.fixed = True
            samples = self.m.sample(num_samples=20, Lmax=10, epsilon=0.05)
            sample_dict = self.m.get_samples_df(samples)

            ls_trace = sample_dict['model.kern.lengthscales']
            assert np.all([np.all(v == ls_trace[0]) for v in ls_trace])


if __name__ == "__main__":
    unittest.main()
