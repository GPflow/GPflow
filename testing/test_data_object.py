from __future__ import print_function
import GPflow
import numpy as np
import unittest
import tensorflow as tf
from GPflow.data_holders import DictData, ScalarData, DataHolderList
import pickle

class TestDataHolderSimple(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.X = DictData(np.random.randn(2, 2), on_shape_change='pass')
        self.m.Y = DictData(np.random.randn(2, 2), on_shape_change='raise')
        self.m.Z = DictData(np.random.randn(2, 2), on_shape_change='recompile')
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


class TestScalarData(unittest.TestCase):
    def setUp(self):
        self.model = GPflow.param.Parameterized()
        self.model.X = ScalarData(10.0)
        self.model.Y = ScalarData(12.0)

    def test_tf_array(self):
        self.assertTrue(isinstance(self.model.X._tf_array, \
                                        tf.python.framework.ops.Tensor))

        with self.model.tf_mode():
            self.assertTrue(isinstance(self.model.X, tf.python.framework.ops.Tensor))

    def test_eval(self):
        feed_dict = self.model.get_feed_dict()
        with self.model.tf_mode():
            value = tf.Session().run(self.model.X*self.model.Y, feed_dict=feed_dict)
        self.assertTrue(np.allclose(self.model.X.value*self.model.Y.value, value))

class TestDataHolderList(unittest.TestCase):
    def setUp(self):
        self.model = GPflow.param.Parameterized()
        self.model.data_list = DataHolderList()
        for i in range(4):
            self.model.data_list.append(DictData(np.random.randn(4,3), on_shape_change='pass'))
        # add a different type tensor
        self.model.data_list.append(DictData(np.random.randint(0,5, (3,3)), on_shape_change='pass'))

    def test_tf_array(self):
        for d in self.model.data_list:
            self.assertTrue(isinstance(d._tf_array, tf.python.framework.ops.Tensor))
            #self.assertTrue(isinstance(d, tf.)
        with self.model.tf_mode():
            data_list_tf = self.model.data_list
            for d in data_list_tf:
                self.assertTrue(isinstance(d, tf.python.framework.ops.Tensor))
            #self.assertTrue(isinstance(d, tf.)


    def test_tf_mode(self):
        with self.model.tf_mode():
            tf_array_list = self.model.data_list
        for tf_array in tf_array_list:
            self.assertTrue(isinstance(tf_array, tf.python.framework.ops.Tensor))

    def test_iter(self):
        for data in self.model.data_list:
            self.assertTrue(isinstance(data, DictData))
        for i in range(len(self.model.data_list)):
            self.assertTrue(isinstance(self.model.data_list[i], DictData))

    """
    def test_str(self):
        # TODO Should be improved if
        string = self.model.data_list
        string = self.model
    """

    def test_pickle(self):
        s = pickle.dumps(self.model.data_list)
        pickle.loads(s)

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


if __name__ == '__main__':
    unittest.main()
