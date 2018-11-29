# Copyright 2016 the GPflow authors
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

# pylint: disable=E1123

import gpflow
import numpy as np
import pandas as pd
import pytest
import tensorflow as tf
from gpflow import GPflowError, settings
from gpflow.test_util import GPflowTestCase, session_tf
from numpy.testing import assert_allclose


class Foo(gpflow.models.Model):
    def _build_likelihood(self):
        return tf.zeros([1], dtype=gpflow.settings.float_type)


class TestNaming(GPflowTestCase):
    def test_index(self):
        index = gpflow.core.parentable.Parentable._read_index() + 1
        with self.test_context():
            def increment_assert(i):
                p = gpflow.Param(1)
                assert p.index.split("-")[-1] == i
            for i in range(index, index + 5):
                increment_assert(str(i))

    def test_standard_name(self):
        with self.test_context():
            p = gpflow.Param(1)
            assert p.name.startswith('Parameter')
            assert p.name == p.pathname

            m = gpflow.params.Parameterized()
            assert m.name.startswith('Parameterized')
            assert m.name == m.pathname

    def test_pathname(self):
        with self.test_context():
            a = gpflow.Param(1)
            b = gpflow.Param(1, name='test_name')
            a_pathname = a.pathname
            b_pathname = b.pathname
            assert a.name != b.name
            assert a_pathname != b_pathname
            assert a_pathname == a.full_name
            assert b_pathname == b.full_name

            m = gpflow.params.Parameterized()
            m.a = a
            m.b = b
            assert m.a.name != m.b.name
            assert m.a.pathname != a_pathname
            assert m.b.pathname != b_pathname
            assert m.a.full_name != a_pathname
            assert m.b.full_name != b_pathname
            assert m.a.pathname.split("/")[0] == m.name
            assert m.b.pathname.split("/")[0] == m.name

class TestType(GPflowTestCase):
    def setUp(self):
        int_type = np.int16
        float_type = np.float16

        test_data = [(1, int_type),
                     (1.0, float_type),
                     ([1], float_type),
                     ([1.0], float_type),
                     (np.array([1, 1], dtype=np.float32), np.float32),
                     (np.array([1, 1], dtype=np.int32), np.int32)]

        self.int_type = int_type
        self.float_type = float_type
        self.test_data = test_data

    def test_specific_dtype(self):
        test_data = self.test_data + [
            (1, np.float32),
            (1.0, np.float64),
            ([1.0], np.float32),
            (np.array([1, 2, 3], dtype=np.float64), np.float16)
        ]
        with self.test_context():
            for v, vtype in test_data:
                p = gpflow.Param(v, dtype=vtype, autobuild=False)
                self.assertEqual(p.dtype, vtype)
                p.compile()
                self.assertEqual(p.dtype, vtype)

    def test_default_type(self):
        s = gpflow.settings.get_settings()
        s.dtypes.int_type = self.int_type
        s.dtypes.float_type = self.float_type

        with gpflow.settings.temp_settings(s), self.test_context():
            for v, vtype in self.test_data:
                p = gpflow.Param(v)
                self.assertEqual(p.dtype, vtype)

    def test_assign_fail_types(self):
        with self.test_context():
            param = gpflow.Param(np.array([1]), dtype=np.int32, autobuild=False)
            def fail_assigns(p):
                with self.assertRaises(ValueError):
                    p.assign([2], dtype=np.float32)
                with self.assertRaises(ValueError):
                    p.assign(np.array([2], dtype=np.float32))
                with self.assertRaises(ValueError):
                    p.assign(np.array([2]), dtype=np.float32)
                with self.assertRaises(ValueError):
                    p.assign([2], dtype=np.int64)
            fail_assigns(param)
            param.compile()
            fail_assigns(param)


class TestParameter(GPflowTestCase):
    def setUp(self):
        with self.test_context():
            self.p = gpflow.Param(1.0)
            self.m = gpflow.params.Parameterized()
            self.m.p = gpflow.Param(1.0)
            self.m.b = gpflow.Param(1.0)

    def test_parameter_different_options(self):
        with self.test_context() as session:
            val = 10.
            a = gpflow.Param(val)
            assert_allclose(a.read_value(), val)
            self.assertEqual(a.size, 1)

            size = 2
            val = [10.] * size
            b = gpflow.Param([10.] * size, fix_shape=False)
            assert_allclose(b.read_value(), val)
            self.assertEqual(b.dtype, np.float64)
            self.assertEqual(b.size, size)

            size = 3
            val = [10] * size
            c = gpflow.Param(val, dtype=np.float16)
            assert_allclose(c.read_value(), val)
            self.assertEqual(c.dtype, np.float16)
            self.assertEqual(c.size, size)

            size = 4
            val = [10.] * size
            d = gpflow.Param(val, trainable=False)
            assert_allclose(d.read_value(), val)
            self.assertEqual(d.trainable, False)
            self.assertEqual(d.size, size)

            size = 5
            val = [10.] * size
            transform = gpflow.transforms.Log1pe()
            e = gpflow.Param(val, transform=transform)
            assert_allclose(e.read_value(), val)
            self.assertEqual(e.size, size)
            unconstrained = transform.backward(np.array(val))
            assert_allclose(session.run(e.unconstrained_tensor), unconstrained)

            size = 6
            val = [10.] * size
            f = gpflow.Param(val, prior=gpflow.priors.Gaussian(1, 2))
            assert_allclose(f.read_value(), val)
            assert_allclose(f.read_value(session), val)
            self.assertEqual(f.size, size)
            self.assertTrue(isinstance(f.prior, gpflow.priors.Gaussian))

    def test_initialized(self):
        with self.test_context() as session1:
            p = gpflow.Param(1.0)
            self.assertTrue(p.is_initialized(session1))
            with self.test_context() as session2:
                self.assertFalse(p.is_initialized(session2))
                with self.test_context() as session3:
                    p = gpflow.Param(1.0, autobuild=False)
                    self.assertFalse(p.is_initialized(session1))
                    self.assertFalse(p.is_initialized(session2))
                    self.assertFalse(p.is_initialized(session3))
                    p.compile()
                    self.assertFalse(p.is_initialized(session1))
                    self.assertFalse(p.is_initialized(session2))
                    self.assertTrue(p.is_initialized(session3))

        def assert_exception(args, fun, exception):
            for arg in args:
                with self.assertRaises(exception, msg="Raise at '{}'".format(arg)):
                    fun(arg)

        with self.test_context():
            assert_exception(['', 'non-tempty', 1.0, None, object()],
                             p.is_initialized, ValueError)

    def test_fail_scenarios(self):
        with self.test_context() as session:
            p = gpflow.Param(1.0)
            values = ['', 'test', 1., object(), None]
            for v in values:
                def value_error(value):
                    return self.assertRaises(ValueError, msg='Raised at "{}"'.format(value))
                with value_error(v):
                    p.set_trainable(v)
                with value_error(v):
                    p.trainable = v
                with value_error(v):
                    p.is_built(v)

            tensor = tf.get_variable('test', shape=())
            tensor_non_trainable = tf.get_variable(
                'test_non_trainable', shape=(), trainable=False)
            p = gpflow.Param(tensor)
            p_non_trainable = gpflow.Param(1.0, trainable=False)

            with self.assertRaises(GPflowError):
                p_non_trainable._check_tensor_trainable(tensor)

            with self.assertRaises(GPflowError):
                p._check_tensor_trainable(tensor_non_trainable)

            with self.assertRaises(GPflowError):
                p.read_value(session=None)

            for v in ['', 'non-empty', 1.0, object()]:
                with self.assertRaises(ValueError):
                    p.read_value(session=v)

            with self.assertRaises(GPflowError):
                p.set_trainable(False)
            with self.assertRaises(GPflowError):
                p.trainable = False

            with self.assertRaises(GPflowError):
                p.set_trainable(True)
            with self.assertRaises(GPflowError):
                p.trainable = True

            values = ['', 'test', 1., object()]
            for v in values:
                with self.assertRaises(ValueError, msg='Raised at "{}"'.format(v)):
                    p.anchor(v)

            with self.assertRaises(tf.errors.FailedPreconditionError):
                p.anchor(session)

            with self.assertRaises(ValueError):
                tensor = tf.get_variable('test1', shape=(), trainable=False)
                gpflow.Param(tensor)

            with self.assertRaises(ValueError):
                tensor = tf.get_variable('test2', shape=())
                gpflow.Param(tensor, trainable=False)

    def test_str(self):
        with self.test_context():
            def check_str(obj, expect_str):
                expect = [e for e in expect_str.format(name=p.name).split(' ') if e != '']
                got = [e for e in str(obj).split(' ') if e != '']
                print(expect)
                print(got)
                self.assertEqual(expect, got)

            p_str = ('               class prior transform  trainable shape  '
                     'fixed_shape value\n{name}  Parameter  None    (none)'
                     '       True    ()         True   1.0')
            p = gpflow.Param(1., name="short")
            check_str(p, p_str)

            d_str = ('                 class shape  fixed_shape value'
                     '\n{name}  DataHolder    ()        False   1.0')
            d = gpflow.DataHolder(1., name="short")
            check_str(d, d_str)

            params_str = ('                     class prior transform  trainable shape'
                          '  fixed_shape value\n{name}/p  Parameter  None'
                          '    (none)       True    ()         True   1.0')
            params = gpflow.Parameterized(name="short")
            params.p = p
            params.d = d
            check_str(params, params_str)

    def test_generators(self):
        with self.test_context():
            self.assertEqual(len(list(self.m.parameters)), 2)
            self.assertEqual(len(list(self.m.data_holders)), 0)
            self.assertEqual(len(list(self.m.params)), 2)

    def test_assign(self):
        with self.test_context(tf.Graph()) as session:
            with self.assertRaises(GPflowError):
                self.p.read_value(session)

        with self.test_context() as session:
            self.p.assign(2.0)
            self.assertEqual(self.p.read_value(), 2.0)
            self.assertEqual(self.p.value, 2.0)

            self.m.p = 2.0
            self.assertEqual(self.m.p.read_value(), 2.0)
            self.assertEqual(self.m.p.value, 2.0)

            self.p.assign(100.0, session=session)
            self.assertEqual(self.p.read_value(session), 100.0)
            self.assertEqual(self.p.value, 100.0)

    def test_assign_tensor(self):
        with self.test_context():
            tensor = tf.get_variable('a', shape=())
            param = gpflow.Param(tensor)
            with self.assertRaises(GPflowError):
                param.assign(10)

    def test_floating_assign(self):
        with self.test_context():
            val = 10.
            p = gpflow.Param(val, fix_shape=False)
            assert_allclose(p.read_value(), val)

            val = [10, 10]
            p.assign(val)
            assert_allclose(p.read_value(), val)

            val = [10, 10, 10]
            p.assign(val)
            assert_allclose(p.read_value(), val)

            val = [[10, 10, 10], [10, 10, 10]]
            p.assign(val)
            assert_allclose(p.read_value(), val)

        with self.test_context():
            val = 10.
            p = gpflow.Param(val)

            val = [10., 10.]
            with self.assertRaises(ValueError):
                p.assign(val)

            val = [[10.]]
            with self.assertRaises(ValueError):
                p.assign(val)

    def test_create_and_replace(self):
        with self.test_context():
            tensor = tf.get_variable('a', shape=()) + 1.0
            param = gpflow.Param(1e3)

            with self.assertRaises(ValueError):
                external_param = gpflow.Param(tensor)

            external_param = gpflow.Param(tensor, trainable=False)
            new_param = gpflow.Param(1.0, name='new_param')

            self.m.b = external_param
            self.assertEqual(self.m.b, external_param)

            p = self.m.p
            self.m.p = param

            assert self.m.p is param
            assert p.name.startswith('Parameter')
            assert p.root is p

            self.m.d = new_param
            assert self.m.d is new_param
            assert self.m.d.pathname == '{name}/d'.format(name=self.m.name)

    def test_assign_with_compile(self):
        with self.test_context():
            self.p.compile()
            self.m.compile()
            self.p.assign(2.0)
            self.m.p = 2.0
            self.assertEqual(self.p.read_value(), 2.0)
            self.assertEqual(self.m.p.read_value(), 2.0)

    def test_root(self):
        self.assertTrue(self.m.p.root is self.m)

    def test_existing_tensor(self):
        with self.test_context():
            _ = tf.get_variable('param/unconstrained', shape=())
            with self.assertRaises(GPflowError):
                p = gpflow.Param(1.0, name='param')

    def test_trainable(self):
        self.assertTrue(self.p.trainable)
        self.p.trainable = False
        self.assertFalse(self.p.trainable)

        self.assertTrue(self.m.trainable)
        self.m.p.trainable = False
        self.assertFalse(self.m.p.trainable)
        self.assertTrue(self.m.trainable)

    def test_trainable_with_compile(self):
        with self.test_context():
            self.p.compile()
            self.m.compile()
            self.assertTrue(self.p.trainable)
            self.p.trainable = False
            self.assertFalse(self.p.trainable)

            self.assertTrue(self.m.trainable)
            self.m.p.trainable = False
            self.assertTrue(self.m.trainable)
            self.assertFalse(self.m.p.trainable)
            _check_trainable_flag(self.m, self.assertTrue, self.assertFalse)

    def test_fixed_shape(self):
        with self.test_context():
            p = gpflow.Param(1., fix_shape=False)
            self.assertFalse(p.fixed_shape)
            self.assertAllEqual(p.shape, ())
            self.assertEqual(p.size, 1)

            p.assign([10., 10.])
            self.assertFalse(p.fixed_shape)
            self.assertAllEqual(p.shape, (2,))
            self.assertEqual(p.size, 2)

            p.fix_shape()
            self.assertTrue(p.fixed_shape)
            self.assertAllEqual(p.shape, (2,))
            self.assertEqual(p.size, 2)
            p.assign(np.zeros(p.shape))

            with self.assertRaises(ValueError):
                p.assign([1.], force=True)
            with self.assertRaises(ValueError):
                p.assign(1., force=True)
            with self.assertRaises(ValueError):
                p.assign(np.zeros((3,3)), force=True)

class TestParameterized(GPflowTestCase):

    @staticmethod
    def create_layout():
        p = gpflow.Parameterized(name='p')
        p.a = gpflow.Param(10.)
        p.b = gpflow.Param(11.)
        p.c = gpflow.Parameterized()
        p.c.d = gpflow.Param(12., fix_shape=False)
        p.c.e = gpflow.DataHolder(13.)
        return p

    def test_is_built(self):
        with self.test_context():
            p = gpflow.Parameterized()
            self.assertTrue(p.is_built_coherence())

            # TODO(@awav): Should it be NO?
            self.assertEqual(p.is_built_coherence(tf.Graph()), gpflow.Build.YES)

            values = [None, "", 1.0, object()]
            for v in values:
                with self.assertRaises(ValueError, msg='Passed value {}'.format(v)):
                    p.is_built(v)

            p.a = gpflow.Param(1.0)
            self.assertEqual(p.is_built_coherence(), gpflow.Build.NO)

            p.compile()
            not_compatible = gpflow.Build.NOT_COMPATIBLE_GRAPH
            self.assertTrue(p.is_built_coherence())
            self.assertEqual(p.is_built(tf.Graph()), not_compatible)

            with self.assertRaises(GPflowError):
                p.is_built_coherence(tf.Graph())
            for v in values:
                with self.assertRaises(ValueError, msg='Passed value "{}"'.format(v)):
                    p.is_built(v)

    def test_anchor(self):
        with self.test_context() as session:
            p = gpflow.Parameterized()
            p.a = gpflow.Param(1.0)
            p.compile()
            with self.assertRaises(ValueError):
                p.anchor(None)
            new_value = 2.0
            p.a.parameter_tensor.load(new_value)
            p.anchor(session)
            assert_allclose(p.a.read_value(), new_value)

    def test_read_values(self):
        def check_values(values, expected_dict, unexpected_dicts):
            self.assertTrue(values == expected_dict)
            for unexpected_dict in unexpected_dicts:
                self.assertFalse(values == unexpected_dict)

        expected_dict = {'p/a': 10., 'p/b': 11., 'p/c/d': 12.}
        unexpected_dicts = [
            {'p': 10., 'p/b': 11., 'p/c/d': 12.},
            {'p/a': 11., 'p/b': 11., 'p/c/d': 12.},
            {'p/a': 11.}
        ]

        with self.test_context() as session:
            session_new = tf.Session(graph=session.graph)
            self.assertNotEqual(session_new, session)
            with session_new.as_default():
                with gpflow.defer_build():
                    p = self.create_layout()
                    values = p.read_values()
                    check_values(values, expected_dict, unexpected_dicts)
                    p.compile()
                    values = p.read_values()
                    check_values(values, expected_dict, unexpected_dicts)
                    with self.assertRaises(tf.errors.FailedPreconditionError):
                        p.read_values(session=session)

        with self.test_context() as session_fail:
            self.assertFalse(session == session_fail)
            with self.assertRaises(tf.errors.FailedPreconditionError):
                p.read_values(session=session_fail)

        with self.test_context() as session_intialize:
            p.initialize(session=session_intialize)
            values = p.read_values(session=session_intialize)
            check_values(values, expected_dict, unexpected_dicts)

        values = p.read_values(session=session_new)
        check_values(values, expected_dict, unexpected_dicts)
        session_new.close()

    def test_parameterized_assign(self):
        with self.test_context():
            ## Create parameterized object inside context
            p = self.create_layout()

            values = p.read_values()
            values['p/b'] = 100.
            values['p/c/d'] = 100.
            p.assign(values)
            assert_allclose(p.a.read_value(), 10)
            assert_allclose(p.b.read_value(), 100)
            assert_allclose(p.c.d.read_value(), 100)
            values = list(map(float, p.read_values().values()))
            self.assertTrue(set(values) == set([10, 100, 100]))

        with self.test_context() as session:
            assign_values = {'p/a': 1e3, 'p/c/d': 1e4}
            p.assign(assign_values, session=session)
            assert_allclose(p.a.read_value(), 1e3)
            assert_allclose(p.b.read_value(), 100)
            assert_allclose(p.c.d.read_value(), 1e4)
            values = list(map(float, p.read_values().values()))
            self.assertTrue(set(values) == set([1e3, 100, 1e4]))

    def test_parameterized_assign_panda(self):
        with self.test_context():
            p = self.create_layout()
            vals1 = [1e2, 1e3, 1e4]
            vals2 = [2e2, 2e3, 2e4]
            self.assertEqual(len(vals1), len(vals2))

            df1 = pd.DataFrame({'p/a': vals1, 'p/c/d': vals1})
            df2 = pd.DataFrame({'p/a': vals2, 'p/c/d': vals2})

            for i in range(len(vals1)):
                df_slice1 = df1.iloc[i]
                p.assign(df_slice1, force=False)
                values = p.read_values()
                for name in df_slice1.index:
                    assert_allclose(df_slice1[name], values[name])

                df_slice2 = df2.iloc[i]
                p.assign(df_slice2, force=True)
                values = p.read_values()
                for name in df_slice2.index:
                    assert_allclose(df_slice2[name], values[name])

    def test_fail_assign(self):
        with self.test_context():
            p = self.create_layout()
            values = [1.0, {'a': 1.0}, None, "", "artem", object()]
            for v in values:
                with self.assertRaises(ValueError):
                    p.assign(v)

            different_shape = {
                    'p/a': np.zeros((10, 1)),
                    'p/b': -1,
                    'p/c/d': -1
                }

            a = p.a.read_value()
            b = p.b.read_value()
            c_d = p.c.d.read_value()
            with self.assertRaises(ValueError):
                p.assign(different_shape)
            assert_allclose(p.a.read_value(), a)
            assert_allclose(p.b.read_value(), b)
            assert_allclose(p.c.d.read_value(), c_d)

    def test_fix_shapes(self):
        with self.test_context():
            def children(p):
                yield from p.parameters
                yield from p.data_holders

            p = self.create_layout()
            self.assertFalse(all([c.fixed_shape for c in children(p)]))
            p.fix_shape()
            self.assertTrue(all([c.fixed_shape for c in children(p)]))

            p = self.create_layout()
            p.fix_shape(parameters=False, data_holders=True)
            self.assertTrue(all([c.fixed_shape for c in p.data_holders]))
            p.fix_shape(parameters=True)
            self.assertTrue(all([c.fixed_shape for c in p.parameters]))
            self.assertTrue(all([c.fixed_shape for c in children(p)]))

    def test_trainables(self):
        with self.test_context():
            p = self.create_layout()
            self.assertTrue(all([c.trainable for c in p.parameters]))
            self.assertTrue(p.trainable)

            p.set_trainable(False)
            self.assertFalse(all([c.trainable for c in p.parameters]))
            self.assertFalse(p.trainable)

            p.set_trainable(True)
            self.assertTrue(all([c.trainable for c in p.parameters]))
            self.assertTrue(p.trainable)

            values = [None, "test", "", 1]

            for v in values:
                with self.assertRaises(ValueError, msg='Caught exception for "{}"'.format(v)):
                    p.set_trainable(v)


class TestParameterizedNoParameters(GPflowTestCase):
    def setUp(self):
        with self.test_context(), gpflow.defer_build():
            self.m = gpflow.params.Parameterized(name='m')
            self.m.p = gpflow.params.Parameterized()
            self.m.b = gpflow.params.Parameterized()

    def test_feeds_empty(self):
        with self.test_context():
            p = gpflow.Parameterized()
            self.assertEqual(p.initializables, [])
            self.assertEqual(p.initializable_feeds, {})
            self.assertEqual(p.feeds, {})

    def test_is_built(self):
        with self.test_context():
            self.assertEqual(self.m.is_built_coherence(), gpflow.Build.YES)

    def test_compile(self):
        with self.test_context():
            self.m.compile()
            self.assertEqual(self.m.is_built_coherence(), gpflow.Build.YES)

    def test_generators(self):
        with self.test_context():
            self.assertEqual(list(self.m.parameters), [])
            self.assertEqual(list(self.m.data_holders), [])
            self.assertEqual(len(list(self.m.params)), 2)

    def test_add_parameter_to_empty_parameterized(self):
        with self.test_context():
            self.m.compile()
            self.m.a = gpflow.Param(10)
            self.assertEqual(self.m.is_built_coherence(), gpflow.Build.NO)
            self.m.compile()
            self.assertEqual(self.m.is_built_coherence(), gpflow.Build.YES)
            with self.assertRaises(GPflowError):
                self.m.b = gpflow.Param(20)


class TestParameterizedCompile(GPflowTestCase):
    def setUp(self):
        self.test_graph = tf.Graph()
        with self.test_context() as session:
            self.graph = session.graph
            tensor = tf.get_variable('a', shape=())
            self.m = gpflow.params.Parameterized(name='m')
            self.m.p = gpflow.params.Parameterized()
            self.m.a = gpflow.Param(tensor)
            self.m.b = gpflow.Param(1.0, trainable=False)
            self.m.c = gpflow.Param(np.array([1.0, 2.0]))
            self.m.p.d = gpflow.Param(1.0)

    def test_compile(self):
        with self.test_context():
            tensor = self.m.a.parameter_tensor
            self.m.compile()
            self.assertEqual(len(list(self.m.parameters)), 4)
            self.assertEqual(len(list(self.m.trainable_tensors)), 3)
            self.assertEqual(self.m.a.parameter_tensor, tensor)
            for param in self.m.parameters:
                self.assertTrue(gpflow.misc.is_tensor(param.parameter_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.constrained_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.prior_tensor))

    def test_modify_compiled(self):
        with self.test_context():
            self.assertEqual(len(list(self.m.parameters)), 4)
            self.assertEqual(len(list(self.m.trainable_tensors)), 3)
            for param in self.m.parameters:
                self.assertTrue(gpflow.misc.is_tensor(param.parameter_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.constrained_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.prior_tensor))

    def test_fails_after_compile(self):
        with self.test_context(self.graph):
            self.m.compile()
            with self.assertRaises(GPflowError):
                self.m.d = gpflow.Param(1.0)
            with self.assertRaises(AttributeError):
                _param = self.m.d

    def test_compile(self):
        with self.test_context():
            self.m.compile()
        with self.test_context() as session:
            self.m.compile(session=session)

class TestAutobuild(GPflowTestCase):
    def test_autobuild_option(self):
        with self.test_context():
            foo = Foo(autobuild=False)
            equal = self.assertEqual
            equal(foo.is_built(tf.get_default_graph()), gpflow.Build.NO)
            equal(foo.is_built_coherence(), gpflow.Build.NO)

            p = gpflow.Param(10)
            equal(p.is_built(tf.get_default_graph()), gpflow.Build.YES)
            equal(p.is_built_coherence(), gpflow.Build.YES)

            b = gpflow.Param(10, autobuild=False)
            equal(b.is_built(tf.get_default_graph()), gpflow.Build.NO)
            equal(b.is_built_coherence(), gpflow.Build.NO)

            foo.p = p
            equal(foo.p, p)
            equal(hasattr(foo, 'p'), True)
            equal(foo.is_built(tf.get_default_graph()), gpflow.Build.NO)
            equal(foo.is_built_coherence(), gpflow.Build.NO)

            foo.b = b
            equal(foo.b, b)
            equal(hasattr(foo, 'b'), True)
            equal(foo.is_built(tf.get_default_graph()), gpflow.Build.NO)
            equal(foo.is_built_coherence(), gpflow.Build.NO)

            foo.compile()
            equal(foo.is_built(tf.get_default_graph()), gpflow.Build.YES)
            equal(foo.is_built_coherence(), gpflow.Build.YES)
            equal(p.is_built(tf.get_default_graph()), gpflow.Build.YES)
            equal(p.is_built_coherence(), gpflow.Build.YES)
            equal(b.is_built(tf.get_default_graph()), gpflow.Build.YES)
            equal(b.is_built_coherence(), gpflow.Build.YES)

class TestParameterizedDeep(GPflowTestCase):
    def setUp(self):
        with self.test_context():
            self.m = gpflow.params.Parameterized(name='m')
            self.m.a = gpflow.Param(1.0, trainable=False)
            self.m.foo = gpflow.params.Parameterized()
            self.m.foo.bar = gpflow.params.Parameterized()
            self.m.foo.bar.baz = gpflow.Param(1.0)

    def test_generators(self):
        with self.test_context():
            self.assertEqual(len(list(self.m.parameters)), 2)
            self.assertEqual(len(list(self.m.data_holders)), 0)
            self.assertEqual(len(list(self.m.params)), 2)

    def test_root(self):
        self.assertTrue(self.m.foo.root is self.m)
        self.assertTrue(self.m.foo.bar.root is self.m)
        self.assertTrue(self.m.foo.bar.baz.root is self.m)

    def test_deep_name(self):
        assert self.m.foo.pathname == 'm/foo'
        assert self.m.foo.bar.pathname == 'm/foo/bar'
        assert self.m.foo.bar.baz.pathname == 'm/foo/bar/baz'

    def test_deep_trainable(self):
        with self.test_context():
            self.m.compile()
            self.m.trainable = False
            self.assertEqual(len(list(self.m.trainable_tensors)), 0)
            _check_trainable_flag(self.m, self.assertTrue, self.assertFalse)
            self.m.trainable = True
            self.assertEqual(
                len(list(self.m.parameters)),
                len(list(self.m.trainable_tensors)))
            _check_trainable_flag(self.m, self.assertTrue, self.assertFalse)


class TestParamLikeInvariant(GPflowTestCase):
    def test_self_reference(self):
        m = gpflow.params.Parameterized()
        with self.assertRaises(ValueError):
            m.foo = m
        m.foo = gpflow.params.Parameterized()
        with self.assertRaises(ValueError):
            m.foo.bar = m

    def test_reassign(self):
        m = gpflow.params.Parameterized()
        p = gpflow.params.Parameterized()
        m.foo = p  # assign
        m.foo = p  # reassign

        # TODO(@awav):
        # m = gpflow.params.Parameterized()
        # m.foo = gpflow.params.Parameterized()
        # m.foo.bar = gpflow.params.Parameterized()
        # with self.assertRaises(ValueError):
        #     m.baz = m.foo.bar

        # TODO(@awav):
        #m = gpflow.params.Parameterized()
        #m.foo = gpflow.params.Parameterized()
        #m.foo.bar = gpflow.params.Parameterized()
        #m.boo = gpflow.params.Parameterized()
        #with self.assertRaises(ValueError):
        #    m.boo.far = m.foo.bar

    # TODO(@awav):
    # def testAddingToAnother(self):
    #     """
    #     Adding the same Paramterized object to another tree is fine.
    #     """
    #     m1 = gpflow.params.Parameterized()
    #     m1.foo = gpflow.params.Parameterized()
    #     m2 = gpflow.params.Parameterized()
    #     with self.assertRaises(GPflowError):
    #         m2.foo = m1.foo


class TestParamList(GPflowTestCase):
    def test_construction(self):
        with self.test_context():
            gpflow.ParamList([])
            gpflow.ParamList([gpflow.Param(1)])
            gpflow.ParamList([1.0, np.array([1, 2]), gpflow.Param(1.0)])
            with self.assertRaises(ValueError):
                gpflow.ParamList([gpflow.Param(1), 'stringsnotallowed'])
            with self.assertRaises(ValueError):
                # tuples not valid in constuctor:
                gpflow.ParamList((gpflow.Param(1),))
            with self.assertRaises(ValueError):
                # param objects not valid in constructor (must be in list)
                gpflow.ParamList(gpflow.Param(1))

            with gpflow.defer_build():
                p = gpflow.ParamList([0.0])
                p[0] = gpflow.Param(1.0)
                with self.assertRaises(ValueError):
                    p[0] = 1.0
                with self.assertRaises(ValueError):
                    p[0] = "test"

            p = gpflow.ParamList([])
            p.append(gpflow.Param(1.0))
            p.append(gpflow.Param(2.0))
            p.append(2.0)
            self.assertEqual(len(p), 3)
            with self.assertRaises(ValueError):
                p.append("test")

    def test_naming(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.float_type))
            l = gpflow.ParamList([p1, p2])
            assert p1.pathname == l.name + '/0'
            assert p2.pathname == l.name + '/1'

    def test_setitem(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.float_type))
            param_list = gpflow.ParamList([p1, p2], name='param_list', autobuild=False)

            self.assertEqual(p1.read_value(), param_list[0].read_value())
            self.assertTrue(np.all(param_list[1].read_value() == p2.read_value()))

            param_list[0] = gpflow.Param(2.0)
            self.assertEqual(p1.read_value(), 1.2)
            self.assertEqual(p1.root, p1)
            self.assertEqual(param_list[0].read_value(), 2.0)

            arr = np.array([1.1, 2.2], settings.float_type)
            param_list[1] = gpflow.Param(arr)
            self.assertEqual(p2.root, p2)
            self.assertTrue(np.all(param_list[1].read_value() == arr))

            param_list.compile()
            with self.assertRaises(GPflowError):
                param_list[0] = gpflow.Param(12)

    def test_append(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p4 = gpflow.Param(np.array([3.4, 5.6], settings.float_type))
            with gpflow.defer_build():
                p2 = gpflow.Param(1.2)
                param_list = gpflow.ParamList([p1])
                param_list.append(p2)
            p3 = gpflow.Param(1.2)
            param_list.append(p3)
            param_list.compile()
            with self.assertRaises(gpflow.GPflowError):
                param_list.append(p4)
            self.assertTrue(p1 in param_list.params)
            self.assertTrue(p2 in param_list.params)
            self.assertTrue(p3 in param_list.params)
            self.assertFalse(p4 in param_list.params)
            with self.assertRaises(ValueError):
                param_list.append('foo')

    def test_len(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.float_type))
            l = gpflow.ParamList([p1, p2])
            self.assertTrue(len(l) == 2)

    def test_with_parameterized(self):
        with self.test_context():
            pzd = gpflow.params.Parameterized()
            p = gpflow.Param(1.2)
            pzd.p = p
            param_list = gpflow.ParamList([pzd])
            param_list[0].p = 5.
            self.assertEqual(param_list[0].p.read_value(), 5)

    def test_in_model(self):
        class Foo(gpflow.models.Model):
            def __init__(self):
                gpflow.models.Model.__init__(self)
                self.param_list = gpflow.ParamList([gpflow.Param(1.), gpflow.Param(12.)])

            @gpflow.params_as_tensors
            def _build_likelihood(self):
                return -tf.add_n([tf.square(x) for x in self.param_list])

        with self.test_context():
            m = Foo()
            m.compile()
            optimizer = gpflow.train.ScipyOptimizer()
            optimizer.minimize(m, maxiter=10)
            atol = 1e-6 if settings.float_type is np.float32 else 1e-8
            params = [param.read_value() for param in m.parameters]
            self.assertTrue(np.allclose(params, 0., atol=atol))


class TestFixWithPrior(GPflowTestCase):
    """
    This tests that models with a fixed parameter which has a prior continue to work
    """

    def test_non_trainable_with_prior(self):
        with self.test_context():
            m = Foo(autobuild=False)
            m.p = gpflow.Param(1.0, gpflow.transforms.positive, autobuild=False)
            m.pp = gpflow.Param(1.0, gpflow.transforms.positive, autobuild=False)
            m.p.prior = gpflow.priors.Gamma(1, 1)
            m.pp.prior = gpflow.priors.Gamma(1, 1)
            m.p.trainable = False
            m.compile()
            optimizer = gpflow.train.ScipyOptimizer()
            optimizer.minimize(m, maxiter=10)

#class TestRandomizeDefault(GPflowTestCase):
#    """
#    This tests that distributions can sample random values without priors
#    """
#
#    def test(self):
#        with self.test_context():
#            np.random.seed(1)
#            m = gpflow.models.Model()
#            m.p = gpflow.Param(1.0)
#            m.pp = gpflow.Param(1.0, gpflow.transforms.Log1pe())
#            m.pf = gpflow.Param(1.0)
#            m.pf.trainable = False
#
#            m.pmd = gpflow.Param(np.ones((5, 2)))
#            ltr = gpflow.transforms.LowerTriangular(1,2).forward(np.ones(2 * 10))
#            m.pmd2 = gpflow.Param(
#                ltr, transform=gpflow.transforms.LowerTriangular(1,2))
#
#            #should work as (pseudo) random vals a.s. are not 1.0
#            m.p.randomize()
#            self.assertFalse(m.p.value == 1.0)
#            m.pp.randomize()
#            self.assertFalse(m.pp.value == 1.0 or m.pp.value <= 0.0)
#
#            #check if fixing works
#            m.pf.randomize()
#            self.assertTrue(m.pf.value == 1.0)
#            m.pf.randomize(skipfixed=False)
#            self.assertFalse(m.pf.value == 1.0)
#
#            #check multidimensional
#            pmd_shape = m.pmd.shape
#            m.pmd.randomize()
#            self.assertFalse(np.any(m.pmd.value == 1.0))
#            self.assertEquals(m.pmd.shape, pmd_shape)
#
#            #check non size-preserving transform
#            pmd2_shape = m.pmd2.shape
#            m.pmd2.randomize()
#            self.assertFalse(np.any(m.pmd2.value == 1.0))
#            self.assertEquals(m.pmd2.shape, pmd2_shape)
#
#class TestRandomizePrior(GPflowTestCase):
#    """
#    This tests that distributions can sample random values from priors
#    """
#
#    def test(self):
#        with self.test_context():
#            np.random.seed(1)
#            from inspect import getargspec
#
#            m = gpflow.models.Model()
#            m.p = gpflow.Param(1.0)
#            m.pmd = gpflow.Param(
#                np.eye(5), transform=gpflow.transforms.DiagMatrix())
#
#            priors = [obj for obj in gpflow.priors.__dict__.values() if
#                      isinstance(obj, type) and
#                      issubclass(obj, gpflow.priors._prior) and
#                      obj is not gpflow.priors._prior]
#
#            with self.assertRaises(NotImplementedError):
#                m.p = 1.0
#                m.p.prior = gpflow.priors._prior()
#                m.p.randomize()
#
#            for prior in priors:
#                signature = getargspec(prior.__init__)
#                params = {}
#                if signature.defaults is not None:
#                    param_names = signature.args[:-len(signature.defaults)]
#                else:
#                    param_names = signature.args
#                for param in param_names:
#                    if param not in params.keys() and param is not 'self':
#                        params[param] = 1.
#
#                m.p = 1.0
#                m.p.prior = prior(**params)
#                m.pmd.prior = prior(**params)
#                m.p.randomize()
#                m.pmd.randomize()
#                self.assertFalse(m.p.value == 1.0)
#                self.assertFalse(np.any(m.pmd.value == np.ones(5)))
#                self.assertTrue(m.pmd.value.shape == (5,5))
#
#
#class TestRandomizeFeedPriors(GPflowTestCase):
#    """
#    Test if standard randomize behavior can be overriden using
#    distributions keyword.
#    """
#
#    def test(self):
#        with self.test_context():
#            np.random.seed(1)
#            m = gpflow.models.Model()
#            m.p = gpflow.Param(1.0)
#            with self.assertRaises(NotImplementedError):
#                m.p.randomize(distributions={m.p: gpflow.priors._prior()})
#            m.p.randomize(distributions={m.p: gpflow.priors.Gaussian(0, 1)})
#            self.assertFalse(m.p.value == 1.0)
#
#
#class TestRandomizeHierarchical(GPflowTestCase):
#    """
#    This tests that models can randomize all contained parameters
#    """
#
#    def test(self):
#        with self.test_context():
#            np.random.seed(1)
#            m = gpflow.models.Model()
#            m.p = gpflow.Param(1.0)
#            m.p2 = gpflow.Param(1.0)
#            m.m = gpflow.models.Model()
#            m.m.p = gpflow.Param(1.0)
#            m.m.p2 = gpflow.Param(1.0)
#
#            m.p2.prior = gpflow.priors.Gaussian(0, 1)
#            m.m.p2.prior = gpflow.priors.Gaussian(0, 1)
#            m.randomize()
#
#            self.assertFalse(m.p.value == 1.0)
#            self.assertFalse(m.p2.value == 1.0)
#            self.assertFalse(m.m.p.value == 1.0)
#            self.assertFalse(m.m.p2.value == 1.0)


class TestScopes(GPflowTestCase):
    def setUp(self):
        with self.test_context() as session:
            self.graph = session.graph
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            Y = rng.randn(10, 1)
            k = gpflow.kernels.RBF(1)
            self.m = gpflow.models.GPR(X, Y, k)
            self.m.compile()

    def test_likelihood_name(self):
        likelihood = self.m.likelihood_tensor
        expected_name = self.m.tf_name_scope + '/likelihood'
        self.assertTrue(likelihood.name.startswith(expected_name))

    def test_kern_name(self):
        with self.test_context(self.graph):
            @gpflow.name_scope('test_kernel')
            @gpflow.params_as_tensors
            def run_kernel(m):
                return m.kern.K(m.X)
            K = run_kernel(self.m)
            self.assertTrue(K.name.startswith('test_kernel/'))

def _check_trainable_flag(m, assert_true, assert_false):
    for param in m.parameters:
        assert_bool = assert_false
        if param.trainable:
            assert_bool = assert_true
        assert_bool(gpflow.misc.is_tensor_trainable(param.parameter_tensor))


@pytest.fixture
def param(session_tf):
    return gpflow.Param(10.)


@pytest.fixture
def params_tree(session_tf):
    p = gpflow.Parameterized()
    p.a = gpflow.Param(1.)
    return p

def failures():
    return [None, 1, "unknown", object()]


@pytest.mark.parametrize('arg', failures())
def test_parentable_childname_failures(params_tree, arg):
    with pytest.raises(ValueError):
        params_tree.childname(arg)


def test_parentable_childname_not_found(param, params_tree):
    with pytest.raises(KeyError):
        params_tree.childname(param)


@pytest.mark.parametrize('arg', failures())
def test_parentable_set_child_failure(params_tree, arg):
    with pytest.raises(ValueError):
        params_tree.set_child('b', arg)
    with pytest.raises(ValueError):
        params_tree.set_child('a', arg)


def test_parentable_unset_child_not_found(params_tree, param):
    with pytest.raises(ValueError):
        params_tree.unset_child('b', param)
    with pytest.raises(ValueError):
        params_tree.unset_child('a', param)


def test_parentable_unset_child_not_found(params_tree, param):
    with pytest.raises(ValueError):
        params_tree.unset_child('b', param)
    with pytest.raises(ValueError):
        params_tree.unset_child('a', param)


@pytest.mark.parametrize('arg', failures()[1:])
def test_parentable_set_parent_failures(param, arg):
    with pytest.raises(ValueError):
        param.set_parent(arg)


def test_parentable_set_parent_self_reference(params_tree):
    with pytest.raises(ValueError):
        params_tree.a.set_parent(params_tree)


def test_as_pandas_table_static(params_tree):
    pt1 = params_tree.as_pandas_table()
    pt2 = params_tree.as_pandas_table()
    assert pt1.equals(pt2)
    params_tree.a = params_tree.a.value + 5.0
    pt3 = params_tree.as_pandas_table()
    assert not pt1.equals(pt3)


if __name__ == '__main__':
    tf.test.main()
