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

import pytest
import numpy as np
import tensorflow as tf

import gpflow
from gpflow import misc
from gpflow.test_util import GPflowTestCase, session_tf


class TestPublicMethods(GPflowTestCase):

    @staticmethod
    def run_case(name, equal, not_equal, fn):
        graph = tf.Graph()
        session = tf.get_default_session()

        equal(fn(name))
        equal(fn(name, index='0'))
        equal(fn(name, graph=session.graph))

        not_equal(fn(name, index='1'))
        not_equal(fn(name, graph=graph))
        not_equal(fn(name, graph=graph, index='0'))
        not_equal(fn(name, graph=graph, index='1'))

    def test_tensor_by_name(self):
        with self.test_context():
            name = 'tensor'
            variable = tf.get_variable(name, shape=())
            self.assertTrue(gpflow.misc.is_initializable_tensor(variable))

            def equal(found):
                self.assertFalse(gpflow.misc.is_initializable_tensor(found))
                self.assertTrue(found.name == variable.name)

            def not_equal(found):
                self.assertEqual(found, None)

            fn = gpflow.misc.get_tensor_by_name

            graph = tf.Graph()
            session = tf.get_default_session()
            fake_name = "foo"

            equal(fn(name))
            equal(fn(name, index='0'))
            equal(fn(name, graph=session.graph))

            not_equal(fn(name, index='1'))
            not_equal(fn(name, graph=graph))
            not_equal(fn(name, graph=graph, index='0'))
            not_equal(fn(name, graph=graph, index='1'))
            not_equal(fn(fake_name))
            not_equal(fn(fake_name, graph=graph))

    def test_variable_by_name(self):
        with self.test_context():
            name = 'variable'
            variable = tf.get_variable(name, shape=())
            self.assertTrue(gpflow.misc.is_initializable_tensor(variable))

            def equal(found):
                self.assertTrue(gpflow.misc.is_initializable_tensor(found))
                self.assertEqual(found, variable)

            def not_equal(found):
                self.assertEqual(found, None)

            fn = gpflow.misc.get_variable_by_name

            graph = tf.Graph()
            session = tf.get_default_session()
            fake_name = "foo"

            equal(fn(name))
            equal(fn(name, graph=session.graph))
            not_equal(fn(name, graph=graph))
            not_equal(fn(fake_name))
            not_equal(fn(fake_name, graph=graph))

    def test_valid_param(self):
        with self.test_context():
            name = 'tensor'
            tensor = tf.get_variable(name, shape=())
            self.assertTrue(gpflow.misc.is_valid_param_value(tensor))
            self.assertTrue(gpflow.misc.is_valid_param_value(1.0))
            self.assertTrue(gpflow.misc.is_valid_param_value(1))
            self.assertTrue(gpflow.misc.is_valid_param_value([1.0]))
            self.assertTrue(gpflow.misc.is_valid_param_value([1.0, 1, 1]))
            self.assertTrue(gpflow.misc.is_valid_param_value([1, 1.0, 1]))
            self.assertTrue(gpflow.misc.is_valid_param_value([[1.0], [1]]))
            self.assertTrue(gpflow.misc.is_valid_param_value(np.array(1)))
            self.assertTrue(gpflow.misc.is_valid_param_value(np.array(1.0)))
            self.assertTrue(gpflow.misc.is_valid_param_value(np.array([[1.0], [1]])))
            self.assertTrue(gpflow.misc.is_valid_param_value([[1.0], np.array(1.0)]))
            self.assertTrue(gpflow.misc.is_valid_param_value([np.array(1.0), [1.0]]))

            self.assertFalse(gpflow.misc.is_valid_param_value([]))
            self.assertFalse(gpflow.misc.is_valid_param_value(["", 1.0]))
            self.assertFalse(gpflow.misc.is_valid_param_value([1.0, ""]))
            self.assertFalse(gpflow.misc.is_valid_param_value(["a", 1.0]))
            self.assertFalse(gpflow.misc.is_valid_param_value([1.0, "a"]))
            self.assertFalse(gpflow.misc.is_valid_param_value([1.0, [1.0]]))
            self.assertFalse(gpflow.misc.is_valid_param_value([[1.0], 1.0]))
            self.assertFalse(gpflow.misc.is_valid_param_value(""))
            self.assertFalse(gpflow.misc.is_valid_param_value("1.0"))
            self.assertFalse(gpflow.misc.is_valid_param_value("[1.0]"))
            self.assertFalse(gpflow.misc.is_valid_param_value("0.1"))
            self.assertFalse(gpflow.misc.is_valid_param_value(None))
            self.assertFalse(gpflow.misc.is_valid_param_value(object()))
            self.assertFalse(gpflow.misc.is_valid_param_value(self))

    def test_remove_trainable(self):
        with self.test_context():
            graph = tf.Graph()
            var1 = tf.get_variable('var1', shape=())
            var2 = tf.get_variable('var2', shape=(), trainable=False)

            with self.assertRaises(ValueError):
                gpflow.misc.remove_from_trainables(var1, graph=graph)

            gpflow.misc.remove_from_trainables(var1)
            with self.assertRaises(ValueError):
                gpflow.misc.remove_from_trainables(var1)

            with self.assertRaises(ValueError):
                gpflow.misc.remove_from_trainables(var2)


def test_leading_transpose(session_tf):
    dims = [1, 2, 3, 4]
    a = tf.zeros(dims)
    b = misc.leading_transpose(a, [..., -1, -2])
    c = misc.leading_transpose(a, [-1, ..., -2])
    d = misc.leading_transpose(a, [-1, -2, ...])
    e = misc.leading_transpose(a, [3, 2, ...])
    f = misc.leading_transpose(a, [3, -2, ...])

    assert len(a.shape) == len(b.shape) == len(c.shape) == len(d.shape)
    assert len(a.shape) == len(e.shape) == len(f.shape)
    assert b.shape[-2:] == [4, 3]
    assert c.shape[0] == 4 and c.shape[-1] == 3
    assert d.shape[:2] == [4, 3]
    assert d.shape == e.shape == f.shape


@pytest.mark.xfail(raises=ValueError)
def test_leading_transpose_fail(session_tf):
    dims = [1, 2, 3, 4]
    a = tf.zeros(dims)
    misc.leading_transpose(a, [-1, -2])