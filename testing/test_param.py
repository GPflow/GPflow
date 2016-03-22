import unittest
import GPflow
import tensorflow as tf
import numpy as np

class ParamTestsScalar(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.param.Parameterized()
        self.m.p = GPflow.param.Param(1.0)
    
    def testAssign(self):
        self.m.p = 2.0
        self.failUnless(isinstance(self.m.p, GPflow.param.Param))
        self.failUnless(self.m.get_free_state() == 2.0)

    def testHighestParent(self):
        self.failUnless(self.m.p.highest_parent is self.m)
    
    def testName(self):
        self.failUnless(self.m.p.name == 'p')
    
    def testMakeTheano(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.failUnless(l == 1)

        l = self.m.p.make_tf_array(x)
        self.failUnless(l == 1)
    
    def testFreeState(self):
        xx = self.m.get_free_state()
        self.failUnless(np.allclose(xx, np.ones(1)))

        y = np.array([34.0])
        self.m.set_state(y)
        self.failUnless(np.allclose(self.m.get_free_state(), y))
        
    def testFixed(self):
        self.m.p.fixed = True
        self.failUnless(len(self.m.get_free_state()) == 0)
        self.failUnless(self.m.make_tf_array(tf.placeholder('float64')) == 0)

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.p.fixed = True
        self.failUnless(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.p.prior = GPflow.priors.Gaussian(0,1)
        self.failUnless(self.m._needs_recompile)

    def testTheanoMode(self):
        x = tf.placeholder('float64')
        l = self.m.make_tf_array(x)
        self.failUnless(isinstance(self.m.p, GPflow.param.Param))
        with self.m.tf_mode():
            self.failUnless(isinstance(self.m.p, tf.python.framework.ops.Tensor))

class ParamTestsDeeper(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.param.Parameterized()
        self.m.foo = GPflow.param.Parameterized()
        self.m.foo.bar = GPflow.param.Parameterized()
        self.m.foo.bar.baz = GPflow.param.Param(1.0)
    
    def testHighestParent(self):
        self.failUnless(self.m.foo.highest_parent is self.m)
        self.failUnless(self.m.foo.bar.highest_parent is self.m)
        self.failUnless(self.m.foo.bar.baz.highest_parent is self.m)
    
    def testName(self):
        self.failUnless(self.m.foo.name == 'foo')
        self.failUnless(self.m.foo.bar.name == 'bar')
        self.failUnless(self.m.foo.bar.baz.name == 'baz')
    
    def testMakeTheano(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.failUnless(l == 1)

        l = self.m.foo.make_tf_array(x)
        self.failUnless(l == 1)
    
        l = self.m.foo.bar.make_tf_array(x)
        self.failUnless(l == 1)
    
        l = self.m.foo.bar.baz.make_tf_array(x)
        self.failUnless(l == 1)
    
    def testFreeState(self):
        xx = self.m.get_free_state()
        self.failUnless(np.allclose(xx, np.ones(1)))

        y = np.array([34.0])
        self.m.set_state(y)
        self.failUnless(np.allclose(self.m.get_free_state(), y))
        
    def testFixed(self):
        self.m.foo.bar.baz.fixed = True
        self.failUnless(len(self.m.get_free_state()) == 0)

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.foo.bar.baz.fixed = True
        self.failUnless(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.foo.bar.baz.prior = GPflow.priors.Gaussian(0,1)
        self.failUnless(self.m._needs_recompile)

    def testTheanoMode(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.failUnless(isinstance(self.m.foo.bar.baz, GPflow.param.Param))
        with self.m.tf_mode():
            self.failUnless(isinstance(self.m.foo.bar.baz, tf.python.framework.ops.Tensor))


class ParamTestswider(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.param.Parameterized()
        self.m.foo = GPflow.param.Param(1.0)
        self.m.bar = GPflow.param.Param(np.arange(10))
        self.m.baz = GPflow.param.Param(np.random.randn(3,3))
    
    def testHighestParent(self):
        self.failUnless(self.m.foo.highest_parent is self.m)
        self.failUnless(self.m.bar.highest_parent is self.m)
        self.failUnless(self.m.baz.highest_parent is self.m)
    
    def testName(self):
        self.failUnless(self.m.foo.name == 'foo')
        self.failUnless(self.m.bar.name == 'bar')
        self.failUnless(self.m.baz.name == 'baz')
    
    def testMakeTheano(self):
        x = tf.placeholder('float64')

        l = self.m.make_tf_array(x)
        self.failUnless(l == 20)

        l = self.m.foo.make_tf_array(x)
        self.failUnless(l == 1)
    
        l = self.m.bar.make_tf_array(x)
        self.failUnless(l == 10)
    
        l = self.m.baz.make_tf_array(x)
        self.failUnless(l == 9)
    
    def testFreeState(self):
        xx = self.m.get_free_state()
        self.failUnless(len(xx) == 20)

        y = np.random.randn(20)
        self.m.set_state(y)
        self.failUnless(np.allclose(self.m.get_free_state(), y))
        
    def testFixed(self):
        self.m.foo.fixed = True
        self.failUnless(len(self.m.get_free_state()) == 19)
        
        self.m.foo.fixed = False
        self.m.bar.fixed = True
        self.failUnless(len(self.m.get_free_state()) == 10)

    def testRecompile(self):
        self.m._needs_recompile = False
        self.m.foo.fixed = True
        self.failUnless(self.m._needs_recompile)

        self.m._needs_recompile = False
        self.m.bar.prior = GPflow.priors.Gaussian(0,1)
        self.failUnless(self.m._needs_recompile)

    def testTheanoMode(self):
        x = tf.placeholder('float64')
        l = self.m.make_tf_array(x)
        self.failUnless(all([isinstance(p, GPflow.param.Param) for p in (self.m.foo, self.m.bar, self.m.baz)]))
        with self.m.tf_mode():
            self.failUnless(all([isinstance(p, tf.python.framework.ops.Tensor) for p in (self.m.foo, self.m.bar, self.m.baz)]))



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
        self.failUnless(p1.name == 'item0')
        self.failUnless(p2.name == 'item1')

    def test_connected(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1, p2])
        x = l.get_free_state()
        x.sort()
        self.failUnless( np.all( x==np.array([1.2, 3.4, 5.6]) ) )


    def test_setitem(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1, p2])

        l[0] = 1.2
        self.failUnless( p1._array == 1.2 )

        l[1] = np.array([1.1, 2.2])
        self.failUnless( np.all(p2._array == np.array([1.1, 2.2])) )

        with self.assertRaises(TypeError):
            l[0] = GPflow.param.Param(12)

    def test_append(self):
        p1 = GPflow.param.Param(1.2)
        p2 = GPflow.param.Param(np.array([3.4, 5.6]))
        l = GPflow.param.ParamList([p1])
        l.append(p2)
        self.failUnless( p2 in l.sorted_params )


        with self.assertRaises(AssertionError):
            l.append('foo')

    def test_with_parameterized(self):
        pzd = GPflow.param.Parameterized()
        p = GPflow.param.Param(1.2)
        pzd.p = p
        l = GPflow.param.ParamList([pzd])

        #test assignment:
        l[0].p = 5
        self.failUnless( l.get_free_state()==5 )


    def test_in_model(self):

        class Foo(GPflow.model.Model):
            def __init__(self):
                GPflow.model.Model.__init__(self)
                self.l = GPflow.param.ParamList([
                    GPflow.param.Param(1), GPflow.param.Param(12)])
            def build_likelihood(self):
                return -reduce(tf.add, [tf.square(x) for x in self.l])

        m = Foo()
        self.failUnless(m.get_free_state().size ==2)
        m.optimize()
        self.failUnless(np.allclose(m.get_free_state(), 0.))







if __name__ == "__main__":
    unittest.main()

