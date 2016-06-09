from functools import reduce
import unittest
import GPflow
import tensorflow as tf
import numpy as np


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

    def testMakeTheano(self):
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

    def testTheanoMode(self):
        x = tf.placeholder('float64')
        l = self.m.make_tf_array(x)
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

    def testMakeTheano(self):
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

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.foo.bar.baz.fixed = True
        self.assertTrue(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.foo.bar.baz.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def testTheanoMode(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.assertTrue(isinstance(self.m.foo.bar.baz, GPflow.param.Param))
        with self.m.tf_mode():
            self.assertTrue(isinstance(self.m.foo.bar.baz, tf.python.framework.ops.Tensor))


class ParamTestswider(unittest.TestCase):
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

    def testMakeTheano(self):
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

    def testFixed(self):
        self.m.foo.fixed = True
        self.assertTrue(len(self.m.get_free_state()) == 19)

        self.m.foo.fixed = False
        self.m.bar.fixed = True
        self.assertTrue(len(self.m.get_free_state()) == 10)

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.foo.fixed = True
        self.assertTrue(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.bar.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def testTheanoMode(self):
        x = tf.placeholder('float64')
        l = self.m.make_tf_array(x)
        self.assertTrue(all([isinstance(p, GPflow.param.Param) for p in (self.m.foo, self.m.bar, self.m.baz)]))
        with self.m.tf_mode():
            self.assertTrue(all([isinstance(p, tf.python.framework.ops.Tensor) for p in (self.m.foo, self.m.bar, self.m.baz)]))


class TestParamList(unittest.TestCase):
    def test_construction(self):
        pl1 = GPflow.param.ParamList([])
        pl2 = GPflow.param.ParamList([GPflow.param.Param(1)])
        with self.assertRaises(AssertionError):
            pl2 = GPflow.param.ParamList([GPflow.param.Param(1), 'stringsnotallowed'])

    def test_naming(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1, p2])
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


if __name__ == "__main__":
    unittest.main()
