# Copyright 2017 the GPflow authors
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
import pandas as pd

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase

from numpy.testing import assert_allclose


class TestDataholder(GPflowTestCase):
    def test_create_dataholder(self):
        with self.test_context():
            shape = (10,)
            d = gpflow.DataHolder(np.ones(shape))
            self.assertAllEqual(d.shape, shape)
            self.assertEqual(d.dtype, np.float64)
            self.assertFalse(d.fixed_shape)
            self.assertFalse(d.trainable)

            shape = (10,)
            d = gpflow.DataHolder(np.ones(shape), dtype=gpflow.settings.float_type)
            self.assertAllEqual(d.shape, shape)
            self.assertEqual(d.dtype, gpflow.settings.float_type)
            self.assertFalse(d.fixed_shape)
            self.assertFalse(d.trainable)

            d = gpflow.DataHolder(1)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.int32)
            self.assertFalse(d.fixed_shape)
            self.assertFalse(d.trainable)

            d = gpflow.DataHolder(1.0)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.float64)
            self.assertFalse(d.fixed_shape)
            self.assertFalse(d.trainable)

            size = 10
            shape = (size,)
            d = gpflow.DataHolder([1.] * size)
            self.assertAllEqual(d.shape, shape)
            self.assertEqual(d.dtype, np.float64)
            self.assertFalse(d.fixed_shape)
            self.assertFalse(d.trainable)

            d = gpflow.DataHolder(1.0, fix_shape=True)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.float64)
            self.assertTrue(d.fixed_shape)
            self.assertFalse(d.trainable)

            var = tf.get_variable('dataholder', shape=(), trainable=False)
            d = gpflow.DataHolder(var)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.float32)
            self.assertTrue(d.fixed_shape)
            self.assertFalse(d.trainable)

            tensor = var + 1
            d = gpflow.DataHolder(tensor)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.float32)
            self.assertTrue(d.fixed_shape)
            self.assertFalse(d.trainable)

    def test_is_built(self):
        with self.test_context():
            d = gpflow.DataHolder(1.0)
            with self.assertRaises(ValueError):
                d.is_built(None)

            with self.assertRaises(gpflow.GPflowError):
                d.is_built_coherence(tf.Graph())

    def test_failed_creation(self):
        with self.test_context():
            tensor = tf.get_variable('dataholder', shape=(1,)),
            values = [
                tensor,
                [1, [1, [1]]],
                None,
                "test",
                object(),
            ]
            for value in values:
                with self.assertRaises(ValueError, msg='Value {}'.format(value)):
                    gpflow.DataHolder(tensor)

    def test_fixed_shape(self):
        with self.test_context():
            p = gpflow.DataHolder(1.)
            assert_allclose(1., 1.)
            self.assertFalse(p.fixed_shape)
            self.assertAllEqual(p.shape, ())

            value = [10., 10.]
            p.assign(value)
            assert_allclose(p.read_value(), value)
            self.assertFalse(p.fixed_shape)
            self.assertAllEqual(p.shape, (2,))

            p.fix_shape()
            assert_allclose(p.read_value(), value)
            self.assertTrue(p.fixed_shape)
            self.assertAllEqual(p.shape, (2,))
            p.assign(np.zeros(p.shape))

            value = np.zeros(p.shape)

            with self.assertRaises(ValueError):
                p.assign([1.], force=True)
            assert_allclose(p.read_value(), value)

            with self.assertRaises(ValueError):
                p.assign(1., force=True)
            assert_allclose(p.read_value(), value)

            with self.assertRaises(ValueError):
                p.assign(np.zeros((3, 3)), force=True)
            assert_allclose(p.read_value(), value)


class TestMinibatch(GPflowTestCase):
    def test_create(self):
        with self.test_context():
            values = [tf.get_variable('test', shape=()), "test", None]
            for v in values:
                with self.assertRaises(ValueError):
                    gpflow.Minibatch(v)

    def test_clear(self):
        with self.test_context() as session:
            length = 10
            seed = 10
            arr = np.random.randn(length, 2)
            m = gpflow.Minibatch(arr, shuffle=False)
            self.assertEqual(m.is_built_coherence(), gpflow.Build.YES)
            self.assertEqual(m.seed, None)
            with self.assertRaises(gpflow.GPflowError):
                m.seed = seed
            self.assertEqual(m.seed, None)
            for i in range(length):
                assert_allclose(m.read_value(session=session), [arr[i]])

            m.clear()
            self.assertEqual(m.seed, None)
            m.seed = seed
            self.assertEqual(m.seed, seed)
            self.assertEqual(m.is_built_coherence(), gpflow.Build.NO)
            self.assertEqual(m.parameter_tensor, None)

    def test_seed(self):
        with self.test_context() as session:
            length = 10
            arr = np.random.randn(length, 2)
            batch_size = 2
            m1 = gpflow.Minibatch(arr, seed=1, batch_size=batch_size)
            m2 = gpflow.Minibatch(arr, seed=1, batch_size=batch_size)

            self.assertEqual(m1.is_built_coherence(), gpflow.Build.YES)
            self.assertEqual(m1.seed, 1)
            with self.assertRaises(gpflow.GPflowError):
                m1.seed = 10

            self.assertEqual(m2.is_built_coherence(), gpflow.Build.YES)
            self.assertEqual(m2.seed, 1)
            with self.assertRaises(gpflow.GPflowError):
                m2.seed = 10

            self.assertEqual(m1.seed, 1)
            self.assertEqual(m2.seed, 1)
            for i in range(length):
                m1_value = m1.read_value(session=session)
                m2_value = m2.read_value(session=session)
                self.assertEqual(m1_value.shape[0], batch_size, msg='Index range "{}"'.format(i))
                self.assertEqual(m2_value.shape[0], batch_size, msg='Index range "{}"'.format(i))
                assert_allclose(m1_value, m2_value)

    def test_change_variable_size(self):
        with self.test_context() as session:
            m = gpflow.Parameterized()
            length = 10
            arr = np.random.randn(length, 2)
            m.X = gpflow.Minibatch(arr, shuffle=False)
            for i in range(length):
                assert_allclose(m.X.read_value(session=session), [arr[i]])

            length = 20
            arr = np.random.randn(length, 2)
            m.X = arr
            for i in range(length):
                assert_allclose(m.X.read_value(session=session), [arr[i]])

    def test_change_batch_size(self):
        with self.test_context() as session:
            length = 10
            arr = np.random.randn(length, 2)
            m = gpflow.Minibatch(arr, shuffle=False)
            for i in range(length):
                assert_allclose(m.read_value(session=session), [arr[i]])

            def check_batch_size(m, length, batch_size):
                self.assertEqual(m.batch_size, batch_size)
                for i in range(length//batch_size):
                    value = m.read_value(session=session)
                    self.assertEqual(value.shape[0], batch_size, msg='Index range "{}"'.format(i))

            batch_size = 2
            m.set_batch_size(batch_size)
            check_batch_size(m, length, batch_size)

            batch_size = 5
            m.batch_size = batch_size
            check_batch_size(m, length, batch_size)

            batch_size = 10
            m.set_batch_size(batch_size)
            check_batch_size(m, length, batch_size)
