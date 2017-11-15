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
# limitations under the License.from __future__ import print_function

# pylint: disable=E1123

import tensorflow as tf
import numpy as np

import gpflow
from gpflow import settings
from gpflow.test_util import GPflowTestCase


class Foo(gpflow.models.Model):
    def _build_likelihood(self):
        return tf.zeros([1], dtype=gpflow.settings.np_float)


class TestNaming(GPflowTestCase):
    def test_standard_name(self):
        p_index = gpflow.core.parentable.Parentable._read_index() + 1
        with self.test_context():
            p = gpflow.Param(1)
            self.assertEqual(p.name, 'Parameter')
            self.assertEqual(p.hidden_name, '{}/Parameter'.format(p_index))

    def test_full_fame(self):
        with self.test_context():
            p1_index = gpflow.core.parentable.Parentable._read_index() + 1
            p1 = gpflow.Param(1)
            p2 = gpflow.Param(1, name='test_name')
            self.assertEqual(p1.hidden_full_name, '{index}/Parameter'.format(index=p1_index))
            self.assertEqual(p1.full_name, 'Parameter')
            self.assertEqual(p2.hidden_full_name, 'test_name')
            self.assertEqual(p2.full_name, 'test_name')
            model_index = gpflow.core.parentable.Parentable._read_index() + 1
            m = gpflow.params.Parameterized()
            m.p = p1
            self.assertEqual(m.hidden_full_name, '{index}/Parameterized'.format(index=model_index))
            self.assertEqual(m.full_name, 'Parameterized')
            self.assertEqual(m.p.hidden_full_name, '{index}/Parameterized/p'.format(index=model_index))
            self.assertEqual(m.p.full_name, 'Parameterized/p')
            self.assertEqual(m.full_name, m.name)
            self.assertEqual(m.p.hidden_full_name, '{}/p'.format(m.hidden_name))
            self.assertEqual(m.p.full_name, '{}/p'.format(m.name))

class TestType(GPflowTestCase):
    def setUp(self):
        int_type = tf.int16
        float_type = tf.float16

        test_data = [(1, int_type.as_numpy_dtype),
                     (1.0, float_type.as_numpy_dtype),
                     ([1], float_type.as_numpy_dtype),
                     ([1.0], float_type.as_numpy_dtype),
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


class TestParam(GPflowTestCase):
    def setUp(self):
        with self.test_context():
            self.p = gpflow.Param(1.0)
            self.m_index = gpflow.core.parentable.Parentable._read_index() + 1
            self.m = gpflow.params.Parameterized()
            self.m.p = gpflow.Param(1.0)
            self.m.b = gpflow.Param(1.0)

    def test_generators(self):
        with self.test_context():
            self.assertEqual(len(list(self.m.parameters)), 2)
            self.assertEqual(len(list(self.m.data_holders)), 0)
            self.assertEqual(len(list(self.m.params)), 2)

    def test_assign(self):
        with self.test_context(tf.Graph()) as session:
            with self.assertRaises(gpflow.GPflowError):
                self.p.read_value(session)

        with self.test_context() as session:
            self.p.assign(2.0)
            self.assertEqual(self.p.read_value(), 2.0)

            self.m.p = 2.0
            self.assertEqual(self.m.p.read_value(), 2.0)

            self.p.assign(100.0, session=session)
            self.assertEqual(self.p.read_value(session), 100.0)

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
            self.assertEqual(self.m.p, param)
            self.assertEqual(p.name, 'Parameter')
            self.assertEqual(p.root, p)

            self.m.d = new_param
            self.assertEqual(self.m.d, new_param)
            self.assertEqual(self.m.d.hidden_full_name,
                             '{index}/{name}/d'
                             .format(index=self.m_index, name=self.m.name))
            self.assertEqual(self.m.d.full_name, '{name}/d'.format(name=self.m.name))

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


class TestParameterizedNoParameters(GPflowTestCase):
    def setUp(self):
        with self.test_context(), gpflow.defer_build():
            self.m = gpflow.params.Parameterized(name='m')
            self.m.p = gpflow.params.Parameterized()
            self.m.b = gpflow.params.Parameterized()

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
            with self.assertRaises(gpflow.GPflowError):
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
            with self.assertRaises(gpflow.GPflowError):
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
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')
        self.assertTrue(self.m.foo.full_name == 'm/foo')
        self.assertTrue(self.m.foo.bar.full_name == 'm/foo/bar')
        self.assertTrue(self.m.foo.bar.baz.full_name == 'm/foo/bar/baz')
        self.assertTrue(self.m.foo.hidden_full_name == 'm/foo')
        self.assertTrue(self.m.foo.bar.hidden_full_name == 'm/foo/bar')
        self.assertTrue(self.m.foo.bar.baz.hidden_full_name == 'm/foo/bar/baz')

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
    #     with self.assertRaises(gpflow.GPflowError):
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

    def test_naming(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
            gpflow.ParamList([p1, p2])
            self.assertEqual(p1.name, 'item0')
            self.assertEqual(p2.name, 'item1')

    def test_setitem(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
            param_list = gpflow.ParamList([p1, p2], name='param_list', autobuild=False)

            self.assertEqual(p1.read_value(), param_list[0].read_value())
            self.assertTrue(np.all(param_list[1].read_value() == p2.read_value()))

            param_list[0] = gpflow.Param(2.0)
            self.assertEqual(p1.read_value(), 1.2)
            self.assertEqual(p1.root, p1)
            self.assertEqual(param_list[0].read_value(), 2.0)

            arr = np.array([1.1, 2.2], settings.np_float)
            param_list[1] = gpflow.Param(arr)
            self.assertEqual(p2.root, p2)
            self.assertTrue(np.all(param_list[1].read_value() == arr))

            param_list.compile()
            with self.assertRaises(gpflow.GPflowError):
                param_list[0] = gpflow.Param(12)

    def test_append(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
            param_list = gpflow.ParamList([p1])
            param_list.append(p2)
            self.assertTrue(p2 in param_list.params)
            with self.assertRaises(ValueError):
                param_list.append('foo')

    def test_len(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
            l = gpflow.ParamList([p1])
            l.append(p2)
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
            atol = 1e-6 if settings.np_float is np.float32 else 1e-8
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
        expected_name = self.m.hidden_name + '/likelihood'
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

if __name__ == '__main__':
    tf.test.main()
