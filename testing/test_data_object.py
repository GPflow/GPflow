from __future__ import print_function
import numpy as np
import unittest
import numpy as np
import tensorflow as tf

import gpflow
from gpflow import settings
from testing.gpflow_testcase import GPflowTestCase

float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64


class TestDataHolderSimple(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = gpflow.model.Model()
            self.rng = np.random.RandomState()
            self.m.X = gpflow.param.DataHolder(self.rng.randn(2, 2), on_shape_change='pass')
            self.m.Y = gpflow.param.DataHolder(self.rng.randn(2, 2), on_shape_change='raise')
            self.m.Z = gpflow.param.DataHolder(self.rng.randn(2, 2), on_shape_change='recompile')
            self.m.make_tf_array(np.empty(0))
            self.m._needs_recompile = False

    def test_types(self):
        assert self.m.X.value.dtype == np_float_type
        assert self.m.Y.value.dtype == np_float_type
        assert self.m.Z.value.dtype == np_float_type

    def test_same_shape(self):
        new_X = self.rng.randn(2, 2)
        self.m.X = new_X
        fd = {}
        self.m.update_feed_dict(self.m.get_feed_dict_keys(), fd)
        assert np.all(fd[self.m.X._tf_array] == new_X)
        self.assertFalse(self.m._needs_recompile)

        new_Y = self.rng.randn(2, 2)
        self.m.Y = new_Y
        fd = {}
        self.m.update_feed_dict(self.m.get_feed_dict_keys(), fd)
        assert np.all(fd[self.m.Y._tf_array] == new_Y)
        self.assertFalse(self.m._needs_recompile)

        new_Z = self.rng.randn(2, 2)
        self.m.Z = new_Z
        fd = {}
        self.m.update_feed_dict(self.m.get_feed_dict_keys(), fd)
        assert np.all(fd[self.m.Z._tf_array] == new_Z)
        self.assertFalse(self.m._needs_recompile)

    def test_pass(self):
        self.m.X = self.rng.randn(3, 3)
        self.assertFalse(self.m._needs_recompile)

    def test_raise(self):
        with self.assertRaises(ValueError):
            self.m.Y = self.rng.randn(3, 3)

    def test_recompile(self):
        self.m.Z = self.rng.randn(3, 3)
        self.assertTrue(self.m._needs_recompile)


class TestDataHolderIntegers(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = gpflow.model.Model()
            self.rng = np.random.RandomState()
            self.m.X = gpflow.param.DataHolder(
                self.rng.randint(0, 10, (2, 2)), on_shape_change='pass')
            self.m.X.make_tf_array()
            self.m._needs_recompile = False

    def test_types(self):
        assert self.m.X.value.dtype == np.int32

    def test_same_shape(self):
        new_X = self.rng.randint(0, 10, (2, 2))
        self.m.X = new_X
        fd = {}
        self.m.update_feed_dict(self.m.get_feed_dict_keys(), fd)
        assert np.all(fd[self.m.X._tf_array] == new_X)
        self.assertFalse(self.m._needs_recompile)


class TestDataHolderModels(GPflowTestCase):
    """
    We make test for Dataholder that enables to reuse model for different data
    with the same shape to the original.  We tested this for the six models.
    """
    def setUp(self):
        with self.test_session():
            self.rng = np.random.RandomState(0)
            self.X = self.rng.rand(20, 1)
            self.Y = np.sin(self.X) + 0.9 * np.cos(self.X*1.6) + self.rng.randn(*self.X.shape) * 0.8
            self.kern = gpflow.kernels.Matern32(1)

    def test_gpr(self):
        with self.test_session():
            m = gpflow.gpr.GPR(self.X, self.Y, self.kern)
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(m._needs_recompile,
                             msg="Recompilation should be avoided for the same shape data")

            m.X = self.rng.randn(30, 1)
            self.assertFalse(m._needs_recompile,
                             msg="For GPR, recompilation should be avoided for the same shape data")

    def test_sgpr(self):
        with self.test_session():
            m = gpflow.sgpr.SGPR(self.X, self.Y, self.kern, Z=self.X[::2])
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should be avoided for the same shape data")

            m.X = self.rng.randn(30, 1)
            self.assertFalse(
                m._needs_recompile,
                msg="For SGPR, recompilation should be avoided for the same shape data")

    def test_gpmc(self):
        with self.test_session():
            m = gpflow.gpmc.GPMC(self.X, self.Y, self.kern, likelihood=gpflow.likelihoods.StudentT())
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should be avoided for the same shape data")

            Xnew = self.rng.randn(30, 1)
            Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew*1.6) + self.rng.randn(*Xnew.shape)
            m.X = Xnew
            m.Y = Ynew
            self.assertTrue(
                m._needs_recompile,
                msg="Recompilation should be necessary for different shape data")
            m.compile()  # make sure compilation is okay for new shapes.

    def test_sgpmc(self):
        with self.test_session():
            m = gpflow.sgpmc.SGPMC(
                self.X, self.Y, self.kern,
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X[::2])
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should be avoided for the same shape data")

            m.X = self.rng.randn(30, 1)

            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should not be necessary for different shape data")

    def test_svgp(self):
        with self.test_session():
            m = gpflow.svgp.SVGP(
                self.X, self.Y, self.kern,
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X[::2])
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should be avoided for the same shape data")

            m.X = self.rng.randn(30, 1)

            self.assertFalse(
                m._needs_recompile,
                msg="For SVGP, recompilation should be avoided for new shape data")

    def test_vgp(self):
        with self.test_session():
            m = gpflow.vgp.VGP(self.X, self.Y, self.kern,
                               likelihood=gpflow.likelihoods.StudentT())
            m.compile()
            m.X = self.rng.randn(*self.X.shape)
            self.assertFalse(
                m._needs_recompile,
                msg="Recompilation should be avoided for the same shape data")

            Xnew = self.rng.randn(30, 1)
            Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew*1.6) + self.rng.randn(*Xnew.shape)
            m.X = Xnew
            m.Y = Ynew
            self.assertTrue(
                m._needs_recompile,
                msg="Recompilation should be necessary for different shape data")
            m.compile()  # make sure compilation is okay for new shapes.


if __name__ == '__main__':
    unittest.main()
