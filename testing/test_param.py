# Copyright 2016 the GPflow authors.
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
import gpflow
import tensorflow as tf
import numpy as np

from testing.gpflow_testcase import GPflowTestCase
from gpflow import settings


float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

try:
    import cPickle as pickle
except ImportError:
    import pickle


class NamingTests(GPflowTestCase):
    def test_unnamed(self):
        p = gpflow.param.Param(1)
        self.assertTrue(p.name == 'unnamed')

    def test_bad_parent(self):
        p = gpflow.param.Param(1)
        m = gpflow.model.Model()
        p._parent = m  # do not do this.
        with self.assertRaises(ValueError):
            print(p.name)


class ParamTestsScalar(GPflowTestCase):
    def setUp(self):
        self.m = gpflow.param.Parameterized()
        self.m.p = gpflow.param.Param(1.0)

    def testAssign(self):
        with self.test_session():
            self.m.p = 2.0
            self.assertTrue(isinstance(self.m.p, gpflow.param.Param))
            self.assertTrue(self.m.get_free_state() == 2.0)

    def testValue(self):
        # make sure the correct value is returned
        self.m.p = 3.0
        self.assertTrue(isinstance(self.m.p.value, np.ndarray))

        # make sure assignment does not work

        with self.assertRaises(AttributeError):
            self.m.p.value = 2.53

        # make sure we get a copy
        self.assertFalse(self.m.p.value is self.m.p._array)

    def testReplacement(self):
        old_p = self.m.p
        new_p = gpflow.param.Param(3.0)
        self.m.p = new_p
        # Parameterized instances should not have _needs_recompile
        self.assertFalse(hasattr(self.m, '_needs_recompile'))
        self.assertFalse(old_p.highest_parent is self.m)

    def testHighestParent(self):
        self.assertTrue(self.m.p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.p.name == 'p')

    def testFixing(self):
        self.m.p.fixed = False
        self.m.fixed = True
        self.assertTrue(self.m.p.fixed)
        self.m.p.fixed = False
        self.assertFalse(self.m.fixed)

    def testFixedFreeState(self):
        with self.test_session():
            self.assertTrue(len(self.m.get_free_state()) == 1)
            self.m.set_state(np.ones(1))
            self.m.fixed = True
            self.assertTrue(len(self.m.get_free_state()) == 0)
            self.m.set_state(np.ones(0))

    def testMakeTF(self):
        with self.test_session():
            x = tf.placeholder('float64')

            l = self.m.make_tf_array(x)
            self.assertTrue(l == 1)

            l = self.m.p.make_tf_array(x)
            self.assertTrue(l == 1)

    def testFreeState(self):
        with self.test_session():
            xx = self.m.get_free_state()
            self.assertTrue(np.allclose(xx, np.ones(1)))

            y = np.array([34.0], np_float_type)
            self.m.set_state(y)
            self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testFixed(self):
        with self.test_session():
            self.m.p.fixed = True
            self.assertTrue(len(self.m.get_free_state()) == 0)
            self.assertTrue(self.m.make_tf_array(tf.placeholder(float_type)) == 0)

    def testRecompile(self):
        with self.test_session():
            self.m._needs_recompile = False
            self.m.p.fixed = True
            self.assertTrue(self.m._needs_recompile)

            self.m._needs_recompile = False
            self.m.p.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        with self.test_session():
            x = tf.placeholder('float64')
            self.m.make_tf_array(x)
            self.assertTrue(isinstance(self.m.p, gpflow.param.Param))
            with self.m.tf_mode():
                self.assertTrue(isinstance(self.m.p, tf.Tensor))


class ParamTestsDeeper(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = gpflow.param.Parameterized()
            self.m.foo = gpflow.param.Parameterized()
            self.m.foo.bar = gpflow.param.Parameterized()
            self.m.foo.bar.baz = gpflow.param.Param(1.0)

    def testHighestParent(self):
        self.assertTrue(self.m.foo.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.baz.highest_parent is self.m)

    def testReplacement(self):
        old_p = self.m.foo.bar.baz
        new_p = gpflow.param.Param(3.0)
        self.m.foo.bar.baz = new_p
        # Parameterized instances should not have _needs_recompile
        self.assertFalse(hasattr(self.m, '_needs_recompile'))
        self.assertFalse(old_p.highest_parent is self.m)

    def testReplacement2(self):
        old_p = self.m.foo.bar
        new_p = gpflow.param.Parameterized()
        new_p.baz = gpflow.param.Param(3.0)
        self.m.foo.bar = new_p
        self.assertTrue(new_p.baz.highest_parent is self.m)
        self.assertFalse(old_p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')

    def testMakeTF(self):
        with self.test_session():
            x = tf.placeholder('float64')

            l = self.m.make_tf_array(x)
            self.assertTrue(l == 1)

            l = self.m.foo.make_tf_array(x)
            self.assertTrue(l == 1)

            l = self.m.foo.bar.make_tf_array(x)
            self.assertTrue(l == 1)

            l = self.m.foo.bar.baz.make_tf_array(x)
            self.assertTrue(l == 1)

    def testFreeState(self):
        with self.test_session():
            xx = self.m.get_free_state()
            self.assertTrue(np.allclose(xx, np.ones(1)))

            y = np.array([34.0], np_float_type)
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
        with self.test_session():
            self.m._needs_recompile = False
            self.m.foo.bar.baz.fixed = True
            self.assertTrue(self.m._needs_recompile)

            self.m._needs_recompile = False
            self.m.foo.bar.baz.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        with self.test_session():
            x = tf.placeholder('float64')

            self.m.make_tf_array(x)
            self.assertTrue(isinstance(self.m.foo.bar.baz, gpflow.param.Param))
            with self.m.tf_mode():
                self.assertTrue(isinstance(self.m.foo.bar.baz, tf.Tensor))


class ParamTestsWider(GPflowTestCase):
    def setUp(self):
        self.m = gpflow.param.Parameterized()
        self.m.foo = gpflow.param.Param(1.0)
        self.m.bar = gpflow.param.Param(np.arange(10))
        self.m.baz = gpflow.param.Param(np.random.randn(3, 3))

    def testHighestParent(self):
        self.assertTrue(self.m.foo.highest_parent is self.m)
        self.assertTrue(self.m.bar.highest_parent is self.m)
        self.assertTrue(self.m.baz.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.bar.name == 'bar')
        self.assertTrue(self.m.baz.name == 'baz')

    def testMakeTF(self):
        with self.test_session():
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
        with self.test_session():
            xx = self.m.get_free_state()
            self.assertTrue(len(xx) == 20)

            y = np.random.randn(20)
            self.m.set_state(y)

            self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testIndexParam(self):
        with self.test_session():
            fs = self.m.get_free_state()
            for p in [self.m.foo, self.m.bar, self.m.baz]:
                index, found = self.m.get_param_index(p)
                self.assertTrue(found)
                self.assertTrue(fs[index] == p.get_free_state()[0])

    def testFixed(self):
        with self.test_session():
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
        with self.test_session():
            self.m._needs_recompile = False
            self.m.foo.fixed = True
            self.assertTrue(self.m._needs_recompile)

            self.m._needs_recompile = False
            self.m.bar.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        with self.test_session():
            x = tf.placeholder('float64')
            self.m.make_tf_array(x)
            self.assertTrue(all([isinstance(p, gpflow.param.Param) for p in (self.m.foo, self.m.bar, self.m.baz)]))
            with self.m.tf_mode():
                self.assertTrue(all([isinstance(p, tf.Tensor)
                                     for p in (self.m.foo, self.m.bar, self.m.baz)]))


class SingleParamterizedInvariantTest(GPflowTestCase):
    """
    Tests that invariants of only allowing a single reference to a given Parameterized in a tree
    """
    def testSelfReference(self):
        """
        Test we raise when a Parameterized object references itself
        """
        m = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.foo = m

    def testReferenceBelow(self):
        """
        Test we raise when we reference the same Parameterized object in a descendent node
        """
        m = gpflow.param.Parameterized()
        m.foo = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.foo.bar = m

    def testReferenceAbove(self):
        """
        Test we raise when we reference the same Parameterized object in an ancestor node
        """
        m = gpflow.param.Parameterized()
        m.foo = gpflow.param.Parameterized()
        m.foo.bar = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.baz = m.foo.bar

    def testReferenceAccross(self):
        """
        Test we raise when we reference the same Parameterized object in a sibling node
        """
        m = gpflow.param.Parameterized()
        m.foo = gpflow.param.Parameterized()
        m.foo.bar = gpflow.param.Parameterized()

        m.boo = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.boo.far = m.foo.bar

    def testAddingToAnother(self):
        """
        Adding the same Paramterized object to another tree is fine.
        """
        m1 = gpflow.param.Parameterized()
        m1.foo = gpflow.param.Parameterized()

        m2 = gpflow.param.Parameterized()
        m2.foo = m1.foo

    def testReassign(self):
        """
        We should be able to reassign the same value to the same param
        """
        m1 = gpflow.param.Parameterized()
        p = gpflow.param.Parameterized()
        m1.foo = p  # assign
        m1.foo = p  # reassign


class SingleParamInvariantTest(GPflowTestCase):
    """
    Tests that invariants of only allowing a single reference to a given Param in a tree
    """
    def testReferenceBelow(self):
        """
        Test we raise when the same Param object is added further down the tree
        """
        m = gpflow.param.Parameterized()
        m.p = gpflow.param.Param(1)
        m.foo = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.foo.p = m.p

    def testReferenceAbove(self):
        """
        Test we raise when we reference the same Param object in a an ancestor node
        """
        m = gpflow.param.Parameterized()
        m.foo = gpflow.param.Parameterized()
        m.foo.p = gpflow.param.Param(1)

        with self.assertRaises(ValueError):
            m.p = m.foo.p

    def testReferenceAccross(self):
        """
        Test we raise when we reference the same Param object in a sibling node
        """
        m = gpflow.param.Parameterized()
        m.foo = gpflow.param.Parameterized()
        m.foo.p = gpflow.param.Param(1)

        m.bar = gpflow.param.Parameterized()

        with self.assertRaises(ValueError):
            m.bar.p = m.foo.p

    def testAddingToAnother(self):
        """
        Adding the same Param object to another tree is fine.
        """
        m1 = gpflow.param.Parameterized()
        m1.foo = gpflow.param.Param(1)

        m2 = gpflow.param.Parameterized()
        m2.foo = m1.foo

    def testReassign(self):
        """
        We should be able to reassign the same value to the same param
        """
        m1 = gpflow.param.Parameterized()
        p = gpflow.param.Param(1)
        m1.foo = p  # assign
        m1.foo = p  # reassign


class TestParamList(GPflowTestCase):
    def test_construction(self):
        gpflow.param.ParamList([])
        gpflow.param.ParamList([gpflow.param.Param(1)])
        with self.assertRaises(AssertionError):
            gpflow.param.ParamList([gpflow.param.Param(1), 'stringsnotallowed'])
        with self.assertRaises(AssertionError):
            # tuples not valid in constuctor:
            gpflow.param.ParamList((gpflow.param.Param(1),))
        with self.assertRaises(AssertionError):
            # param objects not valid in constructor (must be in list)
            gpflow.param.ParamList(gpflow.param.Param(1))

    def test_naming(self):
        p1 = gpflow.param.Param(1.2)
        p2 = gpflow.param.Param(np.array([3.4, 5.6], np_float_type))
        gpflow.param.ParamList([p1, p2])
        self.assertTrue(p1.name == 'item0')
        self.assertTrue(p2.name == 'item1')

    def test_connected(self):
        with self.test_session():
            p1 = gpflow.param.Param(1.2)
            p2 = gpflow.param.Param(np.array([3.4, 5.6], np_float_type))
            l = gpflow.param.ParamList([p1, p2])
            x = l.get_free_state()
            x.sort()
            self.assertTrue(np.all(x == np.array([1.2, 3.4, 5.6], np_float_type)))

    def test_setitem(self):
        p1 = gpflow.param.Param(1.2)
        p2 = gpflow.param.Param(np.array([3.4, 5.6], np_float_type))
        l = gpflow.param.ParamList([p1, p2])

        l[0] = 1.2
        self.assertTrue(p1._array == 1.2)

        l[1] = np.array([1.1, 2.2], np_float_type)
        self.assertTrue(np.all(p2._array == np.array([1.1, 2.2], np_float_type)))

        with self.assertRaises(TypeError):
            l[0] = gpflow.param.Param(12)

    def test_append(self):
        p1 = gpflow.param.Param(1.2)
        p2 = gpflow.param.Param(np.array([3.4, 5.6], np_float_type))
        l = gpflow.param.ParamList([p1])
        l.append(p2)
        self.assertTrue(p2 in l.sorted_params)
        with self.assertRaises(AssertionError):
            l.append('foo')

    def test_len(self):
        p1 = gpflow.param.Param(1.2)
        p2 = gpflow.param.Param(np.array([3.4, 5.6], np_float_type))
        l = gpflow.param.ParamList([p1])
        l.append(p2)
        self.assertTrue(len(l) == 2)

    def test_with_parameterized(self):
        with self.test_session():
            pzd = gpflow.param.Parameterized()
            p = gpflow.param.Param(1.2)
            pzd.p = p
            l = gpflow.param.ParamList([pzd])

            # test assignment:
            l[0].p = 5
            self.assertTrue(l.get_free_state() == 5)

            # test to make sure tf_mode get turned on and off
            self.assertFalse(pzd._tf_mode)
            with l.tf_mode():
                self.assertTrue(pzd._tf_mode)
            self.assertFalse(pzd._tf_mode)

    def test_in_model(self):
        class Foo(gpflow.model.Model):
            def __init__(self):
                gpflow.model.Model.__init__(self)
                self.l = gpflow.param.ParamList([
                    gpflow.param.Param(1), gpflow.param.Param(12)])

            def build_likelihood(self):
                return -reduce(tf.add, [tf.square(x) for x in self.l])

        with self.test_session():
            m = Foo()
            self.assertTrue(m.get_free_state().size == 2)
            m.optimize(disp=False)
            atol = 1e-6 if np_float_type is np.float32 else 1e-8
            self.assertTrue(np.allclose(m.get_free_state(), 0., atol=atol))


class TestPickleAndDict(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            Y = rng.randn(10, 1)
            self.m = gpflow.gpr.GPR(X, Y, kern=gpflow.kernels.RBF(1))

    def test(self):
        # pickle and reload the model
        s1 = pickle.dumps(self.m)
        m1 = pickle.loads(s1)
        d1 = self.m.get_parameter_dict()
        d2 = m1.get_parameter_dict()
        for key, val in d1.items():
            assert np.all(val == d2[key])


class TestDictEmpty(GPflowTestCase):
    def setUp(self):
        self.m = gpflow.model.Model()

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 0)
        self.m.set_parameter_dict(d)


class TestDictSimple(GPflowTestCase):
    def setUp(self):
        self.m = gpflow.model.Model()
        self.m.p1 = gpflow.param.Param(np.random.randn(3, 2))
        self.m.p2 = gpflow.param.Param(np.random.randn(10))

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 2)
        state1 = self.m.get_free_state().copy()
        self.m.set_state(state1 * 0)
        self.m.set_parameter_dict(d)
        self.assertTrue(np.all(state1 == self.m.get_free_state()))


class TestDictSVGP(GPflowTestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        X = self.rng.randn(10, 1)
        Y = self.rng.randn(10, 1)
        Z = self.rng.randn(5, 1)
        self.m = gpflow.svgp.SVGP(
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


class TestFixWithPrior(GPflowTestCase):
    """
    This tests that models with a fixed parameter which has a prior continue to work
    """

    def test(self):
        with self.test_session():
            m = gpflow.model.Model()
            m.p = gpflow.param.Param(1.0, gpflow.transforms.positive)
            m.pp = gpflow.param.Param(1.0, gpflow.transforms.positive)
            m.p.prior = gpflow.priors.Gamma(1, 1)
            m.pp.prior = gpflow.priors.Gamma(1, 1)
            m.p.fixed = True
            m.build_likelihood = lambda: tf.zeros([1], tf.float64)
            m.optimize(disp=1, maxiter=10)

class TestRandomizeDefault(GPflowTestCase):
    """
    This tests that distributions can sample random values without priors
    """

    def test(self):
        with self.test_session():
            np.random.seed(1)
            m = gpflow.model.Model()
            m.p = gpflow.param.Param(1.0)
            m.pp = gpflow.param.Param(1.0, gpflow.transforms.Log1pe())
            m.pf = gpflow.param.Param(1.0)
            m.pf.fixed = True

            m.pmd = gpflow.param.Param(np.ones((5, 2)))
            ltr = gpflow.transforms.LowerTriangular(1,2).forward(np.ones(2 * 10))
            m.pmd2 = gpflow.param.Param(
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

class TestRandomizePrior(GPflowTestCase):
    """
    This tests that distributions can sample random values from priors
    """

    def test(self):
        with self.test_session():
            np.random.seed(1)
            from inspect import getargspec

            m = gpflow.model.Model()
            m.p = gpflow.param.Param(1.0)
            m.pmd = gpflow.param.Param(
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


class TestRandomizeFeedPriors(GPflowTestCase):
    """
    Test if standard randomize behavior can be overriden using
    distributions keyword.
    """

    def test(self):
        with self.test_session():
            np.random.seed(1)
            m = gpflow.model.Model()
            m.p = gpflow.param.Param(1.0)
            with self.assertRaises(NotImplementedError):
                m.p.randomize(distributions={m.p: gpflow.priors.Prior()})
            m.p.randomize(distributions={m.p: gpflow.priors.Gaussian(0, 1)})
            self.assertFalse(m.p.value == 1.0)


class TestRandomizeHierarchical(GPflowTestCase):
    """
    This tests that models can randomize all contained parameters
    """

    def test(self):
        with self.test_session():
            np.random.seed(1)
            m = gpflow.model.Model()
            m.p = gpflow.param.Param(1.0)
            m.p2 = gpflow.param.Param(1.0)
            m.m = gpflow.model.Model()
            m.m.p = gpflow.param.Param(1.0)
            m.m.p2 = gpflow.param.Param(1.0)

            m.p2.prior = gpflow.priors.Gaussian(0, 1)
            m.m.p2.prior = gpflow.priors.Gaussian(0, 1)
            m.randomize()

            self.assertFalse(m.p.value == 1.0)
            self.assertFalse(m.p2.value == 1.0)
            self.assertFalse(m.m.p.value == 1.0)
            self.assertFalse(m.m.p2.value == 1.0)


class TestScopes(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            k = gpflow.kernels.RBF(1)
            Y = rng.randn(10, 1)
            self.m = gpflow.gpr.GPR(X, Y, k)
            self.m.compile()

    def test_likelihood_name(self):
        with self.test_session():
            with self.m.tf_mode():
                l = self.m.build_likelihood()
            expected_name = self.m.name + '.build_likelihood'
            self.assertTrue(expected_name in l.name)

    def test_kern_name(self):
        with self.test_session():
            with self.m.tf_mode():
                K = self.m.kern.K(self.m.X)
            self.assertTrue('kern.K' in K.name)


if __name__ == "__main__":
    unittest.main()
