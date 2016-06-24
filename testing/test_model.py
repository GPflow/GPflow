from __future__ import print_function
import GPflow
import tensorflow as tf
import numpy as np
import unittest


class TestOptimize(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)

        class Quadratic(GPflow.model.Model):
            def __init__(self):
                GPflow.model.Model.__init__(self)
                self.x = GPflow.param.Param(rng.randn(10))

            def build_likelihood(self):
                return -tf.reduce_sum(tf.square(self.x))
        self.m = Quadratic()

    def test_adam(self):
        o = tf.train.AdamOptimizer()
        self.m.optimize(o, max_iters=5000)
        self.failUnless(self.m.x.value.max() < 1e-2)

    def test_lbfgsb(self):
        self.m.optimize(display=False)
        self.failUnless(self.m.x.value.max() < 1e-6)


class TestNeedsRecompile(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.p = GPflow.param.Param(1.0)

    def test_fix(self):
        self.m._needs_recompile = False
        self.m.p.fixed = True
        self.failUnless(self.m._needs_recompile)

    def test_replace_param(self):
        self.m._needs_recompile = False
        new_p = GPflow.param.Param(3.0)
        self.m.p = new_p
        self.failUnless(self.m._needs_recompile)

    def test_set_prior(self):
        self.m._needs_recompile = False
        self.m.p.prior = GPflow.priors.Gaussian(0, 1)
        self.failUnless(self.m._needs_recompile)

    def test_set_transform(self):
        self.m._needs_recompile = False
        self.m.p.transform = GPflow.transforms.Identity()
        self.failUnless(self.m._needs_recompile)


class KeyboardRaiser:
    """
    This wraps a function and makes it raise a KeyboardInterrupt after some number of calls
    """
    def __init__(self, iters_to_raise, f):
        self.iters_to_raise, self.f = iters_to_raise, f
        self.count = 0

    def __call__(self, *a, **kw):
        self.count += 1
        if self.count >= self.iters_to_raise:
            raise KeyboardInterrupt
        return self.f(*a, **kw)


class TestKeyboardCatching(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        X = np.random.randn(1000, 3)
        Y = np.random.randn(1000, 3)
        Z = np.random.randn(100, 3)
        self.m = GPflow.sgpr.SGPR(X, Y, Z=Z, kern=GPflow.kernels.RBF(3))

    def test_optimize_np(self):
        x0 = self.m.get_free_state()
        self.m._compile()
        self.m._objective = KeyboardRaiser(15, self.m._objective)
        self.m.optimize(display=0, max_iters=10000, ftol=0, gtol=0)
        x1 = self.m.get_free_state()
        self.failIf(np.allclose(x0, x1))

    def test_optimize_tf(self):
        x0 = self.m.get_free_state()
        callback = KeyboardRaiser(5, lambda x: None)
        o = tf.train.AdamOptimizer()
        self.m.optimize(o, max_iters=15, callback=callback)
        x1 = self.m.get_free_state()
        self.failIf(np.allclose(x0, x1))


class TestLikelihoodAutoflow(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        X = np.random.randn(1000, 3)
        Y = np.random.randn(1000, 3)
        Z = np.random.randn(100, 3)
        self.m = GPflow.sgpr.SGPR(X, Y, Z=Z, kern=GPflow.kernels.RBF(3))

    def test_lik_and_prior(self):
        l0 = self.m.compute_log_likelihood()
        p0 = self.m.compute_log_prior()
        self.m.kern.variance.prior = GPflow.priors.Gamma(1.4, 1.6)
        l1 = self.m.compute_log_likelihood()
        p1 = self.m.compute_log_prior()

        self.failUnless(p0 == 0.0)
        self.failIf(p0 == p1)
        self.failUnless(l0 == l1)


class TestName(unittest.TestCase):
    def test_name(self):
        m = GPflow.model.Model(name='foo')
        assert m.name == 'foo'


if __name__ == "__main__":
    unittest.main()
