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
# limitations under the License.from __future__ import print_function

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
            d = gpflow.DataHolder(np.ones(shape), dtype=gpflow.settings.np_float)
            self.assertAllEqual(d.shape, shape)
            self.assertEqual(d.dtype, gpflow.settings.np_float)
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

            tensor = tensor + 1
            d = gpflow.DataHolder(tensor)
            self.assertAllEqual(d.shape, ())
            self.assertEqual(d.dtype, np.float32)
            self.assertTrue(d.fixed_shape)
            self.assertFalse(d.trainable)


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
