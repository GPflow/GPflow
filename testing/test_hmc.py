import GPflow
import numpy as np
import unittest

class SampleGaussianTest(unittest.TestCase):
    def setUp(self):
        self.f = lambda x : (0.5*np.sum(np.square(x)), x)
        self.x0 = np.zeros(3)

    def test_mean_cov(self):
        samples = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                x0=self.x0, verbose=False, thin=10, burn=0)
        mean = samples.mean(0)
        cov = np.cov(samples.T)
        self.failUnless(np.allclose(mean, np.zeros(3), 1e-1, 1e-1))
        self.failUnless(np.allclose(cov, np.eye(3), 1e-1, 1e-1))
    
    def test_rng(self):
        samples1 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                x0=self.x0, verbose=False, thin=10, burn=0, RNG=np.random.RandomState(10))

        samples2 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                x0=self.x0, verbose=False, thin=10, burn=0, RNG=np.random.RandomState(10))

        samples3 = GPflow.hmc.sample_HMC(self.f, num_samples=1000, Lmax=20, epsilon=0.05,
                x0=self.x0, verbose=False, thin=10, burn=0, RNG=np.random.RandomState(11))

        self.failUnless(np.all(samples1==samples2))
        self.failIf(np.all(samples1==samples3))

    def test_burn(self):

        samples = GPflow.hmc.sample_HMC(self.f, num_samples=100, Lmax=20, epsilon=0.05,
                x0=self.x0, verbose=False, thin=1, burn=10, RNG=np.random.RandomState(11))

        self.failUnless(samples.shape == (100,3))
        self.failIf(np.all(samples[0] == self.x0))


        
    


if __name__ == "__main__":
    unittest.main()

