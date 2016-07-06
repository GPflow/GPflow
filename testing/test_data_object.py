from __future__ import print_function
import GPflow
import numpy as np
import unittest
import tensorflow as tf


class TestDataHolderSimple(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.X = GPflow.param.DataHolder(np.random.randn(2, 2), on_shape_change='pass')
        self.m.Y = GPflow.param.DataHolder(np.random.randn(2, 2), on_shape_change='raise')
        self.m.Z = GPflow.param.DataHolder(np.random.randn(2, 2), on_shape_change='recompile')
        self.m._needs_recompile = False

    def test_same_shape(self):
        new_X = np.random.randn(2, 2)
        self.m.X = new_X
        assert np.all(self.m.get_feed_dict()[self.m.X._tf_array] == new_X)
        self.assertFalse(self.m._needs_recompile)

        new_Y = np.random.randn(2, 2)
        self.m.Y = new_Y
        assert np.all(self.m.get_feed_dict()[self.m.Y._tf_array] == new_Y)
        self.assertFalse(self.m._needs_recompile)

        new_Z = np.random.randn(2, 2)
        self.m.Z = new_Z
        assert np.all(self.m.get_feed_dict()[self.m.Z._tf_array] == new_Z)
        self.assertFalse(self.m._needs_recompile)

    def test_pass(self):
        self.m.X = np.random.randn(3, 3)
        self.assertFalse(self.m._needs_recompile)

    def test_raise(self):
        with self.assertRaises(ValueError):
            self.m.Y = np.random.randn(3, 3)

    def test_recompile(self):
        self.m.Z = np.random.randn(3, 3)
        self.assertTrue(self.m._needs_recompile)


class TestDataHolderModels(unittest.TestCase):
    """
    We make test for Dataholder that enables to reuse model for different data
    with the same shape to the original.  We tested this for the six models.
    """
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)
        self.X = self.rng.rand(20, 1)*10
        self.Y = np.sin(self.X) + 0.9 * np.cos(self.X*1.6) + self.rng.randn(*self.X.shape) * 0.8
        self.kern = GPflow.kernels.Matern32(1)

    def test_gpr(self):
        m = GPflow.gpr.GPR(self.X, self.Y, self.kern)
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)
        self.assertFalse(m._needs_recompile,
                         msg="For GPR, recompilation should be avoided for the same shape data")

    def test_sgpr(self):
        m = GPflow.sgpr.SGPR(self.X, self.Y, self.kern, Z=self.X[::2])
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)
        self.assertFalse(m._needs_recompile,
                         msg="For SGPR, recompilation should be avoided for the same shape data")

    def test_gpmc(self):
        m = GPflow.gpmc.GPMC(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT())
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        Xnew = np.random.randn(30, 1)
        Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew*1.6) + self.rng.randn(*Xnew.shape) * 0.8
        m.X = Xnew
        m.Y = Ynew
        self.assertTrue(m._needs_recompile,
                        msg="Recompilation should be necessary for different shape data")
        m._compile()  # make sure compilation is okay for new shapes.

    def test_sgpmc(self):
        m = GPflow.sgpmc.SGPMC(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT(), Z=self.X[::2])
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should not be necessary for different shape data")

    def test_svgp(self):
        m = GPflow.svgp.SVGP(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT(), Z=self.X[::2])
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        m.X = np.random.randn(30, 1)

        self.assertFalse(m._needs_recompile,
                         msg="For SVGP, recompilation should be avoided for new shape data")

    def test_vgp(self):
        m = GPflow.vgp.VGP(self.X, self.Y, self.kern, likelihood=GPflow.likelihoods.StudentT())
        m._compile()
        m.X = np.random.randn(*self.X.shape)
        self.assertFalse(m._needs_recompile,
                         msg="Recompilation should be avoided for the same shape data")

        Xnew = np.random.randn(30, 1)
        Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew*1.6) + self.rng.randn(*Xnew.shape) * 0.8
        m.X = Xnew
        m.Y = Ynew
        self.assertTrue(m._needs_recompile,
                        msg="Recompilation should be necessary for different shape data")
        m._compile()  # make sure compilation is okay for new shapes.


class Test_Dict(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        self.X = rng.randn(10, 1)
        self.Y = rng.randn(10, 1)
        
        self.m = GPflow.gpr.GPR(self.X, self.Y, kern=GPflow.kernels.RBF(1))
        m2 = GPflow.gpr.GPR(rng.randn(11, 1), rng.randn(11, 1), kern=GPflow.kernels.RBF(1))
        
        self.data_dict = self.m.get_data_dict() # data.shape= (10,1)
        self.data_dict2 = m2.get_data_dict()    # data.shape= (11,1)

    def test_data_dict(self):
        """
        Check restoration (and aquisition) of data_dict
        """
        rng = np.random.RandomState(0)
        # Prepare another instance
        m2 = GPflow.gpr.GPR(rng.randn(10, 1), rng.randn(10, 1), kern=GPflow.kernels.RBF(1))
        
        # restore data from self.data_dict
        m2.set_data_dict(self.data_dict.copy())
        
        # check self.X == self.m.X.value
        assert np.allclose(self.X, m2.X.value)
        assert np.allclose(self.Y, m2.Y.value)
    
    
    def test_recompile(self):
        """
        Check if recompilation flag raises after the data size changed by
        set_data_dict
        """
        rng = np.random.RandomState(0)
        # test GPR
        m2 = GPflow.gpr.GPR(rng.randn(10, 1), rng.randn(10, 1), kern=GPflow.kernels.RBF(1))
        m2._compile()

        # same shape data        
        m2.set_data_dict(self.data_dict.copy())
        self.assertFalse(m2._needs_recompile)
        # different shape data
        m2.set_data_dict(self.data_dict2.copy())
        self.assertFalse(m2._needs_recompile)
        
        # test VGP, change of the 
        m3 = GPflow.vgp.VGP(X=rng.randn(10, 1), Y=rng.randn(10, 1), \
            kern=GPflow.kernels.RBF(1), likelihood=GPflow.likelihoods.Gaussian())
        m3._compile()
        
        # same shape data        
        m3.set_data_dict(self.data_dict.copy())
        self.assertFalse(m3._needs_recompile)
        # different shape data
        m3.set_data_dict(self.data_dict2.copy())
        self.assertTrue(m3._needs_recompile)
        
        
    def test_setData_at_somewhere(self):        
        """
        Test the acquisition and restoration in other part of the model
        """
        rng = np.random.RandomState(0)
        Z = rng.randn(13,1)
        # test GPR
        m2 = GPflow.gpr.GPR(rng.randn(10, 1), rng.randn(10, 1), kern=GPflow.kernels.RBF(1))
        
        # add data in kernel
        m2.kern.Z = GPflow.param.DataHolder(Z, on_shape_change='pass')
        
        # generate dict
        data_dict = m2.get_data_dict()
        
        # update Z
        m2.kern.Z = GPflow.param.DataHolder(rng.randn(12,1), on_shape_change='pass')
        # restore
        m2.set_data_dict(data_dict.copy())

        self.assertTrue(m2.kern.Z, Z)
        
if __name__ == '__main__':
    unittest.main()
