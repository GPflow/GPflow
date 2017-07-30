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
        self.m.optimize(o, maxiter=5000)
        self.assertTrue(self.m.x.value.max() < 1e-2)

    def test_lbfgsb(self):
        self.m.optimize(disp=False)
        self.assertTrue(self.m.x.value.max() < 1e-6)

    def test_feval_counter(self):
        self.m.compile()
        self.m.num_fevals = 0
        for _ in range(10):
            self.m._objective(self.m.get_free_state())
        self.assertTrue(self.m.num_fevals == 10)


class TestNeedsRecompile(unittest.TestCase):
    def setUp(self):
        self.m = GPflow.model.Model()
        self.m.p = GPflow.param.Param(1.0)

    def test_fix(self):
        self.m._needs_recompile = False
        self.m.p.fixed = True
        self.assertTrue(self.m._needs_recompile)

    def test_replace_param(self):
        self.m._needs_recompile = False
        new_p = GPflow.param.Param(3.0)
        self.m.p = new_p
        self.assertTrue(self.m._needs_recompile)

    def test_set_prior(self):
        self.m._needs_recompile = False
        self.m.p.prior = GPflow.priors.Gaussian(0, 1)
        self.assertTrue(self.m._needs_recompile)

    def test_set_transform(self):
        self.m._needs_recompile = False
        self.m.p.transform = GPflow.transforms.Identity()
        self.assertTrue(self.m._needs_recompile)

    def test_replacement(self):
        m = GPflow.model.Model()
        m.p = GPflow.param.Parameterized()
        m.p.p = GPflow.param.Param(1.0)
        m._needs_recompile = False
        # replace Parameterized
        new_p = GPflow.param.Parameterized()
        new_p.p = GPflow.param.Param(1.0)
        m.p = new_p
        self.assertTrue(m._needs_recompile is True)


class TestSessionGraphArguments(unittest.TestCase):
    class Dummy(GPflow.model.Model):
        def __init__(self):
            super(Dummy, self).__init__(self)
            self.x = GPflow.param.Param(10)

        def build_likelihood(self):
            return tf.negative(tf.reduce_sum(tf.square(self.x)))

    def setUp(self):
        tf.reset_default_graph()

    def test_session_graph_properties(self):
        models = [TestSessionGraphArguments.Dummy() for _ in range(5)]
        m1, m2, m3, m4, m5 = models
        session = tf.Session()
        graph = tf.Graph()
        m1.compile()
        m2.compile(session=session)
        m3.compile(graph=graph)
        with graph.as_default():
            m4.compile()
        m5.compile(session=session, graph=graph)

        sessions = [m.session for m in models]
        sess1, sess2, sess3, sess4, sess5 = sessions
        sessions_set = set(map(str, sessions))
        self.assertEqual(len(sessions_set), 4)
        self.assertEqual(sess2, sess5)
        self.assertEqual(sess1.graph, sess2.graph)
        self.assertEqual(sess3.graph, sess4.graph)
        self.assertEqual(sess2.graph, tf.get_default_graph())
        self.assertEqual(sess3.graph, graph)
        self.assertNotEqual(sess1.graph, sess3.graph)


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
        self.m.compile()
        self.m._objective = KeyboardRaiser(15, self.m._objective)
        self.m.optimize(disp=0, maxiter=10000, ftol=0, gtol=0)
        x1 = self.m.get_free_state()
        self.assertFalse(np.allclose(x0, x1))

    def test_optimize_tf(self):
        x0 = self.m.get_free_state()
        callback = KeyboardRaiser(5, lambda x: None)
        o = tf.train.AdamOptimizer()
        self.m.optimize(o, maxiter=15, callback=callback)
        x1 = self.m.get_free_state()
        self.assertFalse(np.allclose(x0, x1))


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

        self.assertTrue(p0 == 0.0)
        self.assertFalse(p0 == p1)
        self.assertTrue(l0 == l1)


class TestName(unittest.TestCase):
    def test_name(self):
        m = GPflow.model.Model(name='foo')
        self.assertEqual(m.name, 'foo')


if __name__ == "__main__":
    unittest.main()
