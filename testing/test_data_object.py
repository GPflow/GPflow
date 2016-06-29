from __future__ import print_function
import GPflow
import numpy as np
import unittest
import tensorflow as tf


class TestDataHolder(unittest.TestCase):
    """
    We make test for Dataholder that enables to reuse model for different data
    with the same shape to the original.  We tested this for the six models.
    """
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        self.X = rng.rand(20, 1)*10
        self.Y = np.sin(self.X) + 0.9 * np.cos(self.X*1.6) + rng.randn(*self.X.shape) * 0.8
        self.kern = GPflow.kernels.Matern32(1)

    def test_gpr(self):
        m = GPflow.gpr.GPR(self.X, self.Y, self.kern)
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)
        self.assertFalse(m._needs_recompile,
                         msg="For GPR, recompilation should be avoided for the same shape data")

    def test_sgpr(self):
        m = GPflow.sgpr.SGPR(self.X, self.Y, self.kern, X=self.X[::2])
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)
        self.assertFalse(m._needs_recompile,
                         msg="For SGPR, recompilation should be avoided for the same shape data")

    def test_gpmc(self):
        m = GPflow.gpmc.GPMC(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT())
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertTrue(m._needs_recompile,
                        msg="Recompilation should be necessary for the same shape data")

    def test_sgpmc(self):
        m = GPflow.sgpmc.SGPMC(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT(), Z=self.X[::2])
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertTrue(m._needs_recompile,
                        msg="Recompilation should be necessary for different shape data")

    def test_svgp(self):
        m = GPflow.svgp.SVGP(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT(), Z=self.X[::2])
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertFalse(m._needs_recompile,
                         msg="For SVGP, recompilation should be avoided for the same shape data")

    def test_vgp(self):
        m = GPflow.vgp.VGP(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT())
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertTrue(m._needs_recompile,
                        msg="Recompilation should be necessary for different shape data")


if __name__ == '__main__':
    unittest.main()
