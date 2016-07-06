from functools import reduce
import unittest
import GPflow
import tensorflow as tf
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle


class NamingTests(unittest.TestCase):
    def test_unnamed(self):
        p = GPflow.param.Param(1)
        self.assertTrue(p.name == 'unnamed')

    def test_bad_parent(self):
        p = GPflow.param.Param(1)
        m = GPflow.model.Model()
        p._parent = m  # do not do this.
        with self.assertRaises(ValueError):
            print(p.name)

    def test_two_parents(self):
        m = GPflow.model.Model()
        m.p = GPflow.param.Param(1)
        m.p2 = m.p  # do not do this!
        with self.assertRaises(ValueError):
            print(m.p.name)


class ParamTestsScalar(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = GPflow.param.Parameterized()
        self.m.p = GPflow.param.Param(1.0)

    def testAssign(self):
        self.m.p = 2.0
        self.assertTrue(isinstance(self.m.p, GPflow.param.Param))
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
        new_p = GPflow.param.Param(3.0)
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

    def testMakeTF(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.assertTrue(l == 1)

        l = self.m.p.make_tf_array(x)
        self.assertTrue(l == 1)

    def testFreeState(self):
        xx = self.m.get_free_state()
        self.assertTrue(np.allclose(xx, np.ones(1)))

        y = np.array([34.0])
        self.m.set_state(y)
        self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testFixed(self):
        self.m.p.fixed = True
        self.assertTrue(len(self.m.get_free_state()) == 0)
        self.assertTrue(self.m.make_tf_array(tf.placeholder('float64')) == 0)

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.p.fixed = True
        self.assertTrue(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.p.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        x = tf.placeholder('float64')
        self.m.make_tf_array(x)
        self.assertTrue(isinstance(self.m.p, GPflow.param.Param))
        with self.m.tf_mode():
            self.assertTrue(isinstance(self.m.p, tf.python.framework.ops.Tensor))


class ParamTestsDeeper(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = GPflow.param.Parameterized()
        self.m.foo = GPflow.param.Parameterized()
        self.m.foo.bar = GPflow.param.Parameterized()
        self.m.foo.bar.baz = GPflow.param.Param(1.0)

    def testHighestParent(self):
        self.assertTrue(self.m.foo.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.highest_parent is self.m)
        self.assertTrue(self.m.foo.bar.baz.highest_parent is self.m)

    def testReplacement(self):
        old_p = self.m.foo.bar.baz
        new_p = GPflow.param.Param(3.0)
        self.m.foo.bar.baz = new_p
        # Parameterized instances should not have _needs_recompile
        self.assertFalse(hasattr(self.m, '_needs_recompile'))
        self.assertFalse(old_p.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.foo.bar.name == 'bar')
        self.assertTrue(self.m.foo.bar.baz.name == 'baz')

    def testMakeTF(self):
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
        xx = self.m.get_free_state()
        self.assertTrue(np.allclose(xx, np.ones(1)))

        y = np.array([34.0])
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
        self.m._needs_recompile = False
        self.m.foo.bar.baz.fixed = True
        self.assertTrue(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.foo.bar.baz.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        x = tf.placeholder('float64')

        self.m.make_tf_array(x)
        self.assertTrue(isinstance(self.m.foo.bar.baz, GPflow.param.Param))
        with self.m.tf_mode():
            self.assertTrue(isinstance(self.m.foo.bar.baz, tf.python.framework.ops.Tensor))


class ParamTestsWider(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.m = GPflow.param.Parameterized()
        self.m.foo = GPflow.param.Param(1.0)
        self.m.bar = GPflow.param.Param(np.arange(10))
        self.m.baz = GPflow.param.Param(np.random.randn(3, 3))

    def testHighestParent(self):
        self.assertTrue(self.m.foo.highest_parent is self.m)
        self.assertTrue(self.m.bar.highest_parent is self.m)
        self.assertTrue(self.m.baz.highest_parent is self.m)

    def testName(self):
        self.assertTrue(self.m.foo.name == 'foo')
        self.assertTrue(self.m.bar.name == 'bar')
        self.assertTrue(self.m.baz.name == 'baz')

    def testMakeTF(self):
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
        xx = self.m.get_free_state()
        self.assertTrue(len(xx) == 20)

        y = np.random.randn(20)
        self.m.set_state(y)

        self.assertTrue(np.allclose(self.m.get_free_state(), y))

    def testIndexParam(self):
        fs = self.m.get_free_state()
        for p in [self.m.foo, self.m.bar, self.m.baz]:
            index, found = self.m.get_param_index(p)
            self.failUnless(found)
            self.failUnless(fs[index] == p.get_free_state()[0])
        
    def testFixed(self):
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
        self.m._needs_recompile = False
        self.m.foo.fixed = True
        self.assertTrue(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.bar.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def testTFMode(self):
        x = tf.placeholder('float64')
        self.m.make_tf_array(x)
        self.assertTrue(all([isinstance(p, GPflow.param.Param) for p in (self.m.foo, self.m.bar, self.m.baz)]))
        with self.m.tf_mode():
            self.assertTrue(all([isinstance(p, tf.python.framework.ops.Tensor)
                                 for p in (self.m.foo, self.m.bar, self.m.baz)]))


class TestParamList(unittest.TestCase):
    def test_construction(self):
        GPflow.param.ParamList([])
        GPflow.param.ParamList([GPflow.param.Param(1)])
        with self.assertRaises(AssertionError):
            GPflow.param.ParamList([GPflow.param.Param(1), 'stringsnotallowed'])

    def test_naming(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        GPflow.param.ParamList([p1, p2])
        self.assertTrue(p1.name == 'item0')
        self.assertTrue(p2.name == 'item1')

    def test_connected(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1, p2])
        x = l.get_free_state()
        x.sort()
        self.assertTrue(np.all(x == np.array([1.2, 3.4, 5.6])))

    def test_setitem(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1, p2])

        l[0] = 1.2
        self.assertTrue(p1._array == 1.2)

        l[1] = np.array([1.1, 2.2])
        self.assertTrue(np.all(p2._array == np.array([1.1, 2.2])))

        with self.assertRaises(TypeError):
            l[0] = GPflow.param.Param(12)

    def test_append(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1])
        l.append(p2)
        self.assertTrue(p2 in l.sorted_params)

        with self.assertRaises(AssertionError):
            l.append('foo')

    def test_with_parameterized(self):
        pzd = GPflow.param.Parameterized()
        p = GPflow.param.Param(1.2)
        pzd.p = p
        l = GPflow.param.ParamList([pzd])

        # test assignment:
        l[0].p = 5
        self.assertTrue(l.get_free_state() == 5)

        # test to make sure tf_mode get turned on and off
        self.assertFalse(pzd._tf_mode)
        with l.tf_mode():
            self.assertTrue(pzd._tf_mode)
        self.assertFalse(pzd._tf_mode)

    def test_in_model(self):
        class Foo(GPflow.model.Model):
            def __init__(self):
                GPflow.model.Model.__init__(self)
                self.l = GPflow.param.ParamList([
                    GPflow.param.Param(1), GPflow.param.Param(12)])

            def build_likelihood(self):
                return -reduce(tf.add, [tf.square(x) for x in self.l])

        m = Foo()
        self.assertTrue(m.get_free_state().size == 2)
        m.optimize(display=False)
        self.assertTrue(np.allclose(m.get_free_state(), 0.))


class TestPickleAndDict(unittest.TestCase):
    def setUp(self):
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        self.m = GPflow.gpr.GPR(X, Y, kern=GPflow.kernels.RBF(1))

    def test(self):
        # pickle and reload the model
        s1 = pickle.dumps(self.m)
        m1 = pickle.loads(s1)

        d1 = self.m.get_parameter_dict()
        d2 = m1.get_parameter_dict()
        for key, val in d1.items():
            assert np.all(val == d2[key])


class TestDictEmpty(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 0)
        self.m.set_parameter_dict(d)


class TestDictSimple(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.p1 = GPflow.param.Param(np.random.randn(3, 2))
        self.m.p2 = GPflow.param.Param(np.random.randn(10))

    def test(self):
        d = self.m.get_parameter_dict()
        self.assertTrue(len(d.keys()) == 2)
        state1 = self.m.get_free_state().copy()
        self.m.set_state(state1 * 0)
        self.m.set_parameter_dict(d)
        self.assertTrue(np.all(state1 == self.m.get_free_state()))


class TestDictSVGP(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        X = self.rng.randn(10, 1)
        Y = self.rng.randn(10, 1)
        Z = self.rng.randn(5, 1)
        self.m = GPflow.svgp.SVGP(X, Y, Z=Z, likelihood=GPflow.likelihoods.Gaussian(), kern=GPflow.kernels.RBF(1))

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


if __name__ == "__main__":
    unittest.main()
