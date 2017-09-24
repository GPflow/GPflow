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

from functools import reduce
import unittest
import tensorflow as tf
import numpy as np

import gpflow
from gpflow import settings, test_util

try:
    import cPickle as pickle
except ImportError:
    import pickle


class NamingTests(test_util.GPflowTestCase):
    def testStandardName(self):
        p = gpflow.Param(1)
        self.assertTrue(p.name == 'Param')

    def testFullName(self):
        p = gpflow.Param(1)
        self.assertEqual(p.full_name, 'Param')
        m = gpflow.models.Model()
        self.assertEqual(m.p.full_name, 'Model')
        m.p = p
        self.assertEqual(m.p.full_name, 'Model/p')


class ParamTests(test_util.GPflowTestCase):
    def setUp(self):
        self.p = gpflow.Param(1.0)
        self.m = gpflow.params.Parameterized()
        self.m.p = gpflow.Param(1.0)
        self.m.b = gpflow.Param(1.0)

    def testAssign(self):
        with self.test_context():
            self.p.assign(2.0)
            self.assertTrue(self.p.read_value() == 2.0)
            self.m.p = 2.0
            self.assertTrue(self.m.p.read_value() == 2.0)

    def testCreate(self):
        with self.test_context():
            tensor = tf.get_variable('a', shape=()) + 1.0
            param = gpflow.Param(1e3)
            external_param = gpflow.Param(tensor)
            new_param = gpflow.Param(1.0, name='new_param')

            self.m.b = external_param
            self.assertEqual(self.m.b, external_param)

            p = self.m.p
            self.m.p = param
            self.assertEqual(self.m.p, param)
            self.assertEqual(p.root, p)
            self.assertEqual(p.name, 'Param')

            self.m.d = new_param
            self.assertEqual(self.m.d, new_param)
            self.assertEqual(self.m.d.full_name, self.m.name + '/d')

    def testAssignWithCompile(self):
        with self.test_context():
            self.p.compile()
            self.m.compile()
            self.p.assign(2.0)
            self.m.p = 2.0
            self.assertTrue(self.p.read_value() == 2.0)
            self.assertTrue(self.m.p.read_value() == 2.0)

    def testRoot(self):
        self.assertTrue(self.m.p.root is self.m)

    def testTrainable(self):
        self.assertTrue(self.p.trainable)
        self.p.trainable = False
        self.assertFalse(self.p.trainable)

        self.assertTrue(self.m.trainable)
        self.m.p.fixed = False
        self.assertTrue(self.m.trainable)
        self.assertFalse(self.m.p.trainable)

    def testTrainableWithCompile(self):
        self.p.compile()
        self.m.compile()
        self.assertTrue(self.p.trainable)
        self.p.trainable = False
        self.assertFalse(self.p.trainable)

        self.assertTrue(self.m.trainable)
        self.m.p.fixed = False
        self.assertTrue(self.m.trainable)
        self.assertFalse(self.m.p.trainable)

class ParamCompileTests(test_util.GPflowTestCase):
    def setUp(self):
        with self.test_context() as session:
            self.graph = session.graph
            tensor = tf.get_variable('a', shape=(), trainable=False)
            self.m = gpflow.params.Parameterized(name='m')
            self.m.p = gpflow.params.Parameterized()
            self.m.a = gpflow.Param(tensor, trainable=False)
            self.m.b = gpflow.Param(1.0)
            self.m.c = gpflow.Param(np.array([1.0, 2.0]))
            self.m.p.d = gpflow.Param(1.0)

    def testCompile(self):
        with self.test_context(self.graph):
            tensor = self.m.a.var_tensor
            self.m.compile()
            self.assertEqual(len(list(self.m.parameters)), 4)
            self.assertEqual(len(list(self.m.trainable_tensors)), 3)
            self.assertEqual(self.m.a.var_tensor, tensor)
            for param in self.m.parameters:
                self.assertTrue(gpflow.misc.is_tensor(param.var_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.transformed_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.prior_tensor))

    def testModifyCompiled(self):
        with self.test_context(self.graph):
            self.m.compile()
            self.assertEqual(len(list(self.m.parameters)), 4)
            self.assertEqual(len(list(self.m.trainable_tensors)), 3)
            for param in self.m.parameters:
                self.assertTrue(gpflow.misc.is_tensor(param.var_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.transformed_tensor))
                self.assertTrue(gpflow.misc.is_tensor(param.prior_tensor))

    def testFailsAfterCompile(self):
        with self.test_context():
            self.m.compile()
            with self.assertRaises(gpflow.GPflowError):
                self.m.d = gpflow.Param(1.0)
            with self.assertRaises(AttributeError):
                param = self.m.d

    def testFailsAtCompile(self):
        with self.test_context():
            with self.assertRaises(gpflow.GPflowError):
                self.m.p.d.compile()
            with self.assertRaises(gpflow.GPflowError):
                self.m.p.compile()
            with self.assertRaises(gpflow.GPflowError):
                self.m.a.compile()
            with self.assertRaises(gpflow.GPflowError):
                self.m.b.compile()
            with self.assertRaises(gpflow.GPflowError):
                self.m.c.compile()
            self.m.compile()


class ParamTestsDeeper(test_util.GPflowTestCase):
    def setUp(self):
        with self.test_context():
            self.m = gpflow.params.Parameterized()
            self.m.foo = gpflow.params.Parameterized()
            self.m.foo.bar = gpflow.params.Parameterized()
            self.m.foo.bar.baz = gpflow.Param(1.0)

    def testRoot(self):
        self.assertTrue(self.m.foo.root is self.m)
        self.assertTrue(self.m.foo.bar.root is self.m)
        self.assertTrue(self.m.foo.bar.baz.root is self.m)

    def testReplacement(self):
        old_p = self.m.foo.bar.baz
        new_p = gpflow.Param(3.0)
        self.m.foo.bar.baz = new_p
        # Parameterized instances should not have _needs_recompile
        self.assertFalse(hasattr(self.m, '_needs_recompile'))
        self.assertFalse(old_p.root is self.m)

    def testReplacement2(self):
        old_p = self.m.foo.bar
        new_p = gpflow.params.Parameterized()
        new_p.baz = gpflow.Param(3.0)
        self.m.foo.bar = new_p
        self.assertTrue(new_p.baz.root is self.m)
        self.assertFalse(old_p.root is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')

    def testFreeState(self):
        with self.test_context():
            xx = self.m.get_free_state()
            self.assertTrue(np.allclose(xx, np.ones(1)))

            y = np.array([34.0], settings.np_float)
            self.m.set_state(y)
            self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testFixed(self):
        self.m.foo.bar.baz.fixed = True
        self.assertTrue(len(self.m.get_free_state()) == 0)

    def testFixing(self):
        self.m.fixed = False
        self.m.foo.bar.fixed = True
        self.assertTrue(self.m.fixed)
        self.assertTrue(self.m.foo.fixed)
        self.assertTrue(self.m.foo.bar.fixed)
        self.assertTrue(self.m.foo.bar.baz.fixed)
        self.m.foo.bar.baz.fixed = False
        self.assertFalse(self.m.fixed)
        self.assertFalse(self.m.foo.fixed)
        self.assertFalse(self.m.foo.bar.fixed)
        self.assertFalse(self.m.foo.bar.baz.fixed)

    def testRecompile(self):
        with self.test_context():
            self.m._needs_recompile = False
            self.m.foo.bar.baz.fixed = True
            self.assertTrue(self.m._needs_recompile)

            self.m._needs_recompile = False
            self.m.foo.bar.baz.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        with self.test_context():
            x = tf.placeholder('float64')

            self.m.make_tf_array(x)
            self.assertTrue(isinstance(self.m.foo.bar.baz, gpflow.Param))
            with self.m.tf_mode():
                self.assertTrue(isinstance(self.m.foo.bar.baz, tf.Tensor))


class ParamTestsWider(test_util.GPflowTestCase):
    def setUp(self):
        self.m = gpflow.params.Parameterized()
        self.m.foo = gpflow.Param(1.0)
        self.m.bar = gpflow.Param(np.arange(10))
        self.m.baz = gpflow.Param(np.random.randn(3, 3))

    def testHighestParent(self):
        self.assertTrue(self.m.foo.root is self.m)
        self.assertTrue(self.m.bar.root is self.m)
        self.assertTrue(self.m.baz.root is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.bar.name == 'bar')
        self.assertTrue(self.m.baz.name == 'baz')

    def testMakeTF(self):
        with self.test_context():
            x = tf.placeholder('float64')

            l = self.m.make_tf_array(x)
            self.assertTrue(l == 20)

            l = self.m.foo.make_tf_array(x)
            self.assertTrue(l == 1)

            l = self.m.bar.make_tf_array(x)
            self.assertTrue(l == 10)

            l = self.m.baz.make_tf_array(x)
            self.assertTrue(l == 9)

    def testFreeState(self):
        with self.test_context():
            xx = self.m.get_free_state()
            self.assertTrue(len(xx) == 20)

            y = np.random.randn(20)
            self.m.set_state(y)

            self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testIndexParam(self):
        with self.test_context():
            fs = self.m.get_free_state()
            for p in [self.m.foo, self.m.bar, self.m.baz]:
                index, found = self.m.get_param_index(p)
                self.assertTrue(found)
                self.assertTrue(fs[index] == p.get_free_state()[0])

    def testFixed(self):
        with self.test_context():
            self.m.foo.fixed = True
            self.assertTrue(len(self.m.get_free_state()) == 19)

            self.m.foo.fixed = False
            self.m.bar.fixed = True
            self.assertTrue(len(self.m.get_free_state()) == 10)

    def testFixing(self):
        self.m.fixed = False
        self.m.foo.fixed = True
        self.assertFalse(self.m.fixed)
        self.assertTrue(self.m.foo.fixed)
        self.assertFalse(self.m.bar.fixed)
        self.assertFalse(self.m.baz.fixed)
        self.m.bar.fixed = True
        self.m.baz.fixed = True
        self.assertTrue(self.m.fixed)
        self.assertTrue(self.m.foo.fixed)
        self.assertTrue(self.m.bar.fixed)
        self.assertTrue(self.m.baz.fixed)

    def testRecompile(self):
        with self.test_context():
            self.m._needs_recompile = False
            self.m.foo.fixed = True
            self.assertTrue(self.m._needs_recompile)

            self.m._needs_recompile = False
            self.m.bar.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        with self.test_context():
            x = tf.placeholder('float64')
            self.m.make_tf_array(x)
            self.assertTrue(all([isinstance(p, gpflow.Param) for p in (self.m.foo, self.m.bar, self.m.baz)]))
            with self.m.tf_mode():
                self.assertTrue(all([isinstance(p, tf.Tensor)
                                     for p in (self.m.foo, self.m.bar, self.m.baz)]))


class SingleParamterizedInvariantTest(test_util.GPflowTestCase):
    """
    Tests that invariants of only allowing a single reference to a given Parameterized in a tree
    """
    def testSelfReference(self):
        """
        Test we raise when a Parameterized object references itself
        """
        m = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.foo = m

    def testReferenceBelow(self):
        """
        Test we raise when we reference the same Parameterized object in a descendent node
        """
        m = gpflow.params.Parameterized()
        m.foo = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.foo.bar = m

    def testReferenceAbove(self):
        """
        Test we raise when we reference the same Parameterized object in an ancestor node
        """
        m = gpflow.params.Parameterized()
        m.foo = gpflow.params.Parameterized()
        m.foo.bar = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.baz = m.foo.bar

    def testReferenceAccross(self):
        """
        Test we raise when we reference the same Parameterized object in a sibling node
        """
        m = gpflow.params.Parameterized()
        m.foo = gpflow.params.Parameterized()
        m.foo.bar = gpflow.params.Parameterized()

        m.boo = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.boo.far = m.foo.bar

    def testAddingToAnother(self):
        """
        Adding the same Paramterized object to another tree is fine.
        """
        m1 = gpflow.params.Parameterized()
        m1.foo = gpflow.params.Parameterized()

        m2 = gpflow.params.Parameterized()
        m2.foo = m1.foo

    def testReassign(self):
        """
        We should be able to reassign the same value to the same param
        """
        m1 = gpflow.params.Parameterized()
        p = gpflow.params.Parameterized()
        m1.foo = p  # assign
        m1.foo = p  # reassign


class SingleParamInvariantTest(test_util.GPflowTestCase):
    """
    Tests that invariants of only allowing a single reference to a given Param in a tree
    """
    def testReferenceBelow(self):
        """
        Test we raise when the same Param object is added further down the tree
        """
        m = gpflow.params.Parameterized()
        m.p = gpflow.Param(1)
        m.foo = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.foo.p = m.p

    def testReferenceAbove(self):
        """
        Test we raise when we reference the same Param object in a an ancestor node
        """
        m = gpflow.params.Parameterized()
        m.foo = gpflow.params.Parameterized()
        m.foo.p = gpflow.Param(1)

        with self.assertRaises(ValueError):
            m.p = m.foo.p

    def testReferenceAccross(self):
        """
        Test we raise when we reference the same Param object in a sibling node
        """
        m = gpflow.params.Parameterized()
        m.foo = gpflow.params.Parameterized()
        m.foo.p = gpflow.Param(1)

        m.bar = gpflow.params.Parameterized()

        with self.assertRaises(ValueError):
            m.bar.p = m.foo.p

    def testAddingToAnother(self):
        """
        Adding the same Param object to another tree is fine.
        """
        m1 = gpflow.params.Parameterized()
        m1.foo = gpflow.Param(1)

        m2 = gpflow.params.Parameterized()
        m2.foo = m1.foo

    def testReassign(self):
        """
        We should be able to reassign the same value to the same param
        """
        m1 = gpflow.params.Parameterized()
        p = gpflow.Param(1)
        m1.foo = p  # assign
        m1.foo = p  # reassign


class TestParamList(test_util.GPflowTestCase):
    def test_construction(self):
        gpflow.ParamList([])
        gpflow.ParamList([gpflow.Param(1)])
        with self.assertRaises(AssertionError):
            gpflow.ParamList([gpflow.Param(1), 'stringsnotallowed'])
        with self.assertRaises(AssertionError):
            # tuples not valid in constuctor:
            gpflow.ParamList((gpflow.Param(1),))
        with self.assertRaises(AssertionError):
            # param objects not valid in constructor (must be in list)
            gpflow.ParamList(gpflow.Param(1))

    def test_naming(self):
        p1 = gpflow.Param(1.2)
        p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
        gpflow.ParamList([p1, p2])
        self.assertTrue(p1.name == 'item0')
        self.assertTrue(p2.name == 'item1')

    def test_connected(self):
        with self.test_context():
            p1 = gpflow.Param(1.2)
            p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
            l = gpflow.ParamList([p1, p2])
            x = l.get_free_state()
            x.sort()
            self.assertTrue(np.all(x == np.array([1.2, 3.4, 5.6], settings.np_float)))

    def test_setitem(self):
        p1 = gpflow.Param(1.2)
        p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
        l = gpflow.ParamList([p1, p2])

        l[0] = 1.2
        self.assertTrue(p1._array == 1.2)

        l[1] = np.array([1.1, 2.2], settings.np_float)
        self.assertTrue(np.all(p2._array == np.array([1.1, 2.2], settings.np_float)))

        with self.assertRaises(TypeError):
            l[0] = gpflow.Param(12)

    def test_append(self):
        p1 = gpflow.Param(1.2)
        p2 = gpflow.Param(np.array([3.4, 5.6], settings.np_float))
        l = gpflow.ParamList([p1])
        l.append(p2)
        self.assertTrue(p2 in l.sorted_params)
        with self.assertRaises(AssertionError):
            l.append('foo')

    def test_len(self):
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
            l = gpflow.ParamList([pzd])

            # test assignment:
            l[0].p = 5
            self.assertTrue(l.get_free_state() == 5)

            # test to make sure tf_mode get turned on and off
            self.assertFalse(pzd._tf_mode)
            with l.tf_mode():
                self.assertTrue(pzd._tf_mode)
            self.assertFalse(pzd._tf_mode)

    def test_in_model(self):
        class Foo(gpflow.models.Model):
            def __init__(self):
                gpflow.models.Model.__init__(self)
                self.l = gpflow.ParamList([
                    gpflow.Param(1), gpflow.Param(12)])

            def build_likelihood(self):
                return -reduce(tf.add, [tf.square(x) for x in self.l])

        with self.test_context():
            m = Foo()
            self.assertTrue(m.get_free_state().size == 2)
            m.optimize(disp=False)
            atol = 1e-6 if settings.np_float is np.float32 else 1e-8
            self.assertTrue(np.allclose(m.get_free_state(), 0., atol=atol))


class TestPickleAndDict(test_util.GPflowTestCase):
    def setUp(self):
        with self.test_context():
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            Y = rng.randn(10, 1)
            self.m = gpflow.models.GPR(X, Y, kern=gpflow.kernels.RBF(1))

    def test(self):
        # pickle and reload the model
        s1 = pickle.dumps(self.m)
        m1 = pickle.loads(s1)
        d1 = self.m.get_parameter_dict()
        d2 = m1.get_parameter_dict()
        for key, val in d1.items():
            assert np.all(val == d2[key])


class TestDictEmpty(test_util.GPflowTestCase):
    def setUp(self):
        self.m = gpflow.models.Model()

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 0)
        self.m.set_parameter_dict(d)


class TestDictSimple(test_util.GPflowTestCase):
    def setUp(self):
        self.m = gpflow.models.Model()
        self.m.p1 = gpflow.Param(np.random.randn(3, 2))
        self.m.p2 = gpflow.Param(np.random.randn(10))

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 2)
        state1 = self.m.get_free_state().copy()
        self.m.set_state(state1 * 0)
        self.m.set_parameter_dict(d)
        self.assertTrue(np.all(state1 == self.m.get_free_state()))


class TestDictSVGP(test_util.GPflowTestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        X = self.rng.randn(10, 1)
        Y = self.rng.randn(10, 1)
        Z = self.rng.randn(5, 1)
        self.m = gpflow.models.SVGP(
            X, Y, Z=Z,
            likelihood=gpflow.likelihoods.Gaussian(),
            kern=gpflow.kernels.RBF(1))

    def test(self):
        loglik1 = self.m.compute_log_likelihood()
        d = self.m.get_parameter_dict()

        # muck up the model
        self.m.set_state(self.rng.randn(self.m.get_free_state().size))
        loglik2 = self.m.compute_log_likelihood()

        # reset the model
        self.m.set_parameter_dict(d)
        loglik3 = self.m.compute_log_likelihood()

        self.assertFalse(np.allclose(loglik1, loglik2))
        self.assertTrue(np.allclose(loglik1, loglik3))


class TestFixWithPrior(test_util.GPflowTestCase):
    """
    This tests that models with a fixed parameter which has a prior continue to work
    """

    def test(self):
        with self.test_context():
            m = gpflow.models.Model()
            m.p = gpflow.Param(1.0, gpflow.transforms.positive)
            m.pp = gpflow.Param(1.0, gpflow.transforms.positive)
            m.p.prior = gpflow.priors.Gamma(1, 1)
            m.pp.prior = gpflow.priors.Gamma(1, 1)
            m.p.fixed = True
            m.build_likelihood = lambda: tf.zeros([1], tf.float64)
            m.optimize(disp=1, maxiter=10)

class TestRandomizeDefault(test_util.GPflowTestCase):
    """
    This tests that distributions can sample random values without priors
    """

    def test(self):
        with self.test_context():
            np.random.seed(1)
            m = gpflow.models.Model()
            m.p = gpflow.Param(1.0)
            m.pp = gpflow.Param(1.0, gpflow.transforms.Log1pe())
            m.pf = gpflow.Param(1.0)
            m.pf.fixed = True

            m.pmd = gpflow.Param(np.ones((5, 2)))
            ltr = gpflow.transforms.LowerTriangular(1,2).forward(np.ones(2 * 10))
            m.pmd2 = gpflow.Param(
                ltr, transform=gpflow.transforms.LowerTriangular(1,2))

            #should work as (pseudo) random vals a.s. are not 1.0
            m.p.randomize()
            self.assertFalse(m.p.value == 1.0)
            m.pp.randomize()
            self.assertFalse(m.pp.value == 1.0 or m.pp.value <= 0.0)

            #check if fixing works
            m.pf.randomize()
            self.assertTrue(m.pf.value == 1.0)
            m.pf.randomize(skipfixed=False)
            self.assertFalse(m.pf.value == 1.0)

            #check multidimensional
            pmd_shape = m.pmd.shape
            m.pmd.randomize()
            self.assertFalse(np.any(m.pmd.value == 1.0))
            self.assertEquals(m.pmd.shape, pmd_shape)

            #check non size-preserving transform
            pmd2_shape = m.pmd2.shape
            m.pmd2.randomize()
            self.assertFalse(np.any(m.pmd2.value == 1.0))
            self.assertEquals(m.pmd2.shape, pmd2_shape)

class TestRandomizePrior(test_util.GPflowTestCase):
    """
    This tests that distributions can sample random values from priors
    """

    def test(self):
        with self.test_context():
            np.random.seed(1)
            from inspect import getargspec

            m = gpflow.models.Model()
            m.p = gpflow.Param(1.0)
            m.pmd = gpflow.Param(
                np.eye(5), transform=gpflow.transforms.DiagMatrix())

            priors = [obj for obj in gpflow.priors.__dict__.values() if
                      isinstance(obj, type) and
                      issubclass(obj, gpflow.priors.Prior) and
                      obj is not gpflow.priors.Prior]

            with self.assertRaises(NotImplementedError):
                m.p = 1.0
                m.p.prior = gpflow.priors.Prior()
                m.p.randomize()

            for prior in priors:
                signature = getargspec(prior.__init__)
                params = {}
                if signature.defaults is not None:
                    param_names = signature.args[:-len(signature.defaults)]
                else:
                    param_names = signature.args
                for param in param_names:
                    if param not in params.keys() and param is not 'self':
                        params[param] = 1.

                m.p = 1.0
                m.p.prior = prior(**params)
                m.pmd.prior = prior(**params)
                m.p.randomize()
                m.pmd.randomize()
                self.assertFalse(m.p.value == 1.0)
                self.assertFalse(np.any(m.pmd.value == np.ones(5)))
                self.assertTrue(m.pmd.value.shape == (5,5))


class TestRandomizeFeedPriors(test_util.GPflowTestCase):
    """
    Test if standard randomize behavior can be overriden using
    distributions keyword.
    """

    def test(self):
        with self.test_context():
            np.random.seed(1)
            m = gpflow.models.Model()
            m.p = gpflow.Param(1.0)
            with self.assertRaises(NotImplementedError):
                m.p.randomize(distributions={m.p: gpflow.priors.Prior()})
            m.p.randomize(distributions={m.p: gpflow.priors.Gaussian(0, 1)})
            self.assertFalse(m.p.value == 1.0)


class TestRandomizeHierarchical(test_util.GPflowTestCase):
    """
    This tests that models can randomize all contained parameters
    """

    def test(self):
        with self.test_context():
            np.random.seed(1)
            m = gpflow.models.Model()
            m.p = gpflow.Param(1.0)
            m.p2 = gpflow.Param(1.0)
            m.m = gpflow.models.Model()
            m.m.p = gpflow.Param(1.0)
            m.m.p2 = gpflow.Param(1.0)

            m.p2.prior = gpflow.priors.Gaussian(0, 1)
            m.m.p2.prior = gpflow.priors.Gaussian(0, 1)
            m.randomize()

            self.assertFalse(m.p.value == 1.0)
            self.assertFalse(m.p2.value == 1.0)
            self.assertFalse(m.m.p.value == 1.0)
            self.assertFalse(m.m.p2.value == 1.0)


class TestScopes(test_util.GPflowTestCase):
    def setUp(self):
        with self.test_context():
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            k = gpflow.kernels.RBF(1)
            Y = rng.randn(10, 1)
            self.m = gpflow.models.GPR(X, Y, k)
            self.m.compile()

    def test_likelihood_name(self):
        with self.test_context():
            with self.m.tf_mode():
                l = self.m.build_likelihood()
            expected_name = self.m.name + '.build_likelihood'
            self.assertTrue(expected_name in l.name)

    def test_kern_name(self):
        with self.test_context():
            with self.m.tf_mode():
                K = self.m.kern.K(self.m.X)
            self.assertTrue('kern.K' in K.name)


if __name__ == "__main__":
    unittest.main()
