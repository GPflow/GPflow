import unittest
import GPflow
import numpy as np
import tensorflow as tf


class PriorModeTests(unittest.TestCase):
    """
    these tests optimize the prior to find the mode numerically. Make sure the
    mode is the same as the known mode.
    """
    def setUp(self):
        tf.reset_default_graph()

        class FlatModel(GPflow.model.Model):
            def build_likelihood(self):
                return 0
        self.m = FlatModel()

    def testGaussianMode(self):
        self.m.x = GPflow.param.Param(1.0)
        self.m.x.prior = GPflow.priors.Gaussian(3, 1)
        self.m.optimize(display=0)

        xmax = self.m.get_free_state()
        self.failUnless(np.allclose(xmax, 3))

    def testGaussianModeMatrix(self):
        self.m.x = GPflow.param.Param(np.random.randn(4, 4))
        self.m.x.prior = GPflow.priors.Gaussian(-1, 10)
        self.m.optimize(display=0)

        xmax = self.m.get_free_state()
        self.failUnless(np.allclose(xmax, -1))

    def testGammaMode(self):
        self.m.x = GPflow.param.Param(1.0)
        shape, scale = 4., 5.
        self.m.x.prior = GPflow.priors.Gamma(shape, scale)
        self.m.optimize(display=0)

        true_mode = (shape - 1.) * scale
        self.failUnless(np.allclose(self.m.x.value, true_mode, 1e-3))

    def testLaplaceMode(self):
        self.m.x = GPflow.param.Param(1.0)
        self.m.x.prior = GPflow.priors.Laplace(3, 10)
        self.m.optimize(display=0)

        xmax = self.m.get_free_state()
        self.failUnless(np.allclose(xmax, 3))

    def testLogNormalMode(self):
        self.m.x = GPflow.param.Param(1.0)
        self.m.x.prior = GPflow.priors.LogNormal(3, 10)
        self.m.x.transform = GPflow.transforms.Exp()
        self.m.optimize(display=0)

        xmax = self.m.get_free_state()
        self.failUnless(np.allclose(xmax, 3))




if __name__ == "__main__":
    unittest.main()
