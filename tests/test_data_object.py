# Copyright 2017 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf

import numpy as np
from numpy.testing import assert_array_equal

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase


class Foo(gpflow.models.Model):
    def _build_likelihood(self):
        return tf.constant(0, dtype=gpflow.settings.float_type)

class TestDataHolderSimple(GPflowTestCase):
    def prepare(self, autobuild=True):
        rng = np.random.RandomState()
        with gpflow.defer_build():
            m = Foo()
            m.X = gpflow.DataHolder(rng.randn(2, 2))
            m.Y = gpflow.DataHolder(rng.randn(2, 2))
            m.Z = gpflow.DataHolder(rng.randn(2, 2))
        if autobuild:
            m.compile()
        return m, rng

    def test_types(self):
        with self.test_context():
            m, _ = self.prepare(False)
            self.assertEqual(m.X.dtype, settings.float_type)
            self.assertEqual(m.Y.dtype, settings.float_type)
            self.assertEqual(m.Z.dtype, settings.float_type)
            self.assertEqual(m.X.read_value().dtype, settings.float_type)
            self.assertEqual(m.Y.read_value().dtype, settings.float_type)
            self.assertEqual(m.Z.read_value().dtype, settings.float_type)
            m.compile()
            self.assertEqual(m.X.dtype, settings.float_type)
            self.assertEqual(m.Y.dtype, settings.float_type)
            self.assertEqual(m.Z.dtype, settings.float_type)
            self.assertEqual(m.X.read_value().dtype, settings.float_type)
            self.assertEqual(m.Y.read_value().dtype, settings.float_type)
            self.assertEqual(m.Z.read_value().dtype, settings.float_type)

    def test_same_shape(self):
        with self.test_context():
            m, rng = self.prepare()

            new_X = rng.randn(2, 2)
            m.X = new_X
            assert_array_equal(m.X.shape, new_X.shape)
            assert_array_equal(m.X.read_value(), new_X)

            new_Y = rng.randn(2, 2)
            m.Y = new_Y
            assert_array_equal(m.Y.shape, new_Y.shape)
            assert_array_equal(m.Y.read_value(), new_Y)

            new_Z = rng.randn(2, 2)
            m.Z = new_Z
            assert_array_equal(m.Z.shape, new_Z.shape)
            assert_array_equal(m.Z.read_value(), new_Z)

    def test_raise(self):
        with self.test_context():
            m, rng = self.prepare(True)
            m.Y = rng.randn(3, 3)


class TestDataHolderIntegers(GPflowTestCase):
    def prepare(self):
        m = Foo(autobuild=False)
        rng = np.random.RandomState()
        m.X = gpflow.DataHolder(rng.randint(0, 10, (2, 2)), dtype=gpflow.settings.int_type)
        m.compile()
        return m, rng

    def test_types(self):
        with self.test_context():
            m, _ = self.prepare()
            self.assertEqual(m.X.read_value().dtype, gpflow.settings.int_type)
            self.assertEqual(m.X.dtype, gpflow.settings.int_type)

    def test_same_shape(self):
        with self.test_context():
            m, rng = self.prepare()
            new_X = rng.randint(0, 10, (2, 2), dtype=gpflow.settings.int_type)
            m.X = new_X
            assert_array_equal(m.X.shape, new_X.shape)
            assert_array_equal(m.X.read_value(), new_X)


class TestDataHolderModels(GPflowTestCase):
    """
    We make test for Dataholder that enables to reuse model for different data
    with the same shape to the original.  We tested this for the six models.
    """
    def prepare(self):
        rng = np.random.RandomState(0)
        X = rng.rand(20, 1)
        Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
        kern = gpflow.kernels.Matern32(1)
        return X, Y, kern, rng

    def test_gpr(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.GPR(X, Y, kern)
            m.compile()
            m.X = rng.randn(*X.shape)
            m.X = rng.randn(30, 1)

    def test_sgpr(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.SGPR(X, Y, kern, Z=X[::2])
            m.compile()
            m.X = rng.randn(*X.shape)
            m.X = rng.randn(30, 1)

    def test_gpmc(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.GPMC(X, Y, kern, likelihood=gpflow.likelihoods.StudentT())
            m.compile()
            m.X = rng.randn(*X.shape)
            Xnew = rng.randn(30, 1)
            Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew*1.6) + rng.randn(*Xnew.shape)
            m.X = Xnew
            m.Y = Ynew

    def test_sgpmc(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.SGPMC(X, Y, kern, likelihood=gpflow.likelihoods.StudentT(), Z=X[::2])
            m.compile()
            m.X = rng.randn(*X.shape)
            m.X = rng.randn(30, 1)

    def test_svgp(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.SVGP(X, Y, kern, likelihood=gpflow.likelihoods.StudentT(), Z=X[::2])
            m.compile()
            m.X = rng.randn(*X.shape)
            m.X = rng.randn(30, 1)

    def test_vgp(self):
        with self.test_context():
            X, Y, kern, rng = self.prepare()
            m = gpflow.models.VGP(X, Y, kern, likelihood=gpflow.likelihoods.StudentT())
            m.compile()
            m.X = rng.randn(*X.shape)
            Xnew = rng.randn(30, 1)
            Ynew = np.sin(Xnew) + 0.9 * np.cos(Xnew * 1.6) + rng.randn(*Xnew.shape)
            m.X = Xnew
            m.Y = Ynew


if __name__ == '__main__':
    tf.test.main()
