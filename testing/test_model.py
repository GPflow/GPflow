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
import gpflow
import tensorflow as tf
import numpy as np
import unittest

from testing.gpflow_testcase import GPflowTestCase


class TestOptimize(GPflowTestCase):
    def setUp(self):
        rng = np.random.RandomState(0)

        class Quadratic(gpflow.model.Model):
            def __init__(self):
                gpflow.model.Model.__init__(self)
                self.x = gpflow.param.Param(rng.randn(10))

            def build_likelihood(self):
                return -tf.reduce_sum(tf.square(self.x))

        self.m = Quadratic()

    def test_adam(self):
        with self.test_session():
            o = tf.train.AdamOptimizer()
            self.m.optimize(o, maxiter=5000)
            self.assertTrue(self.m.x.value.max() < 1e-2)

    def test_lbfgsb(self):
        with self.test_session():
            self.m.optimize(disp=False)
            self.assertTrue(self.m.x.value.max() < 1e-6)

    def test_feval_counter(self):
        with self.test_session():
            self.m.compile()
            self.m.num_fevals = 0
            for _ in range(10):
                self.m._objective(self.m.get_free_state())
            self.assertTrue(self.m.num_fevals == 10)


class TestNeedsRecompile(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            self.m = gpflow.model.Model()
            self.m.p = gpflow.param.Param(1.0)

    def test_fix(self):
        with self.test_session():
            self.m._needs_recompile = False
            self.m.p.fixed = True
            self.assertTrue(self.m._needs_recompile)

    def test_replace_param(self):
        with self.test_session():
            self.m._needs_recompile = False
            new_p = gpflow.param.Param(3.0)
            self.m.p = new_p
            self.assertTrue(self.m._needs_recompile)

    def test_set_prior(self):
        with self.test_session():
            self.m._needs_recompile = False
            self.m.p.prior = gpflow.priors.Gaussian(0, 1)
            self.assertTrue(self.m._needs_recompile)

    def test_set_transform(self):
        with self.test_session():
            self.m._needs_recompile = False
            self.m.p.transform = gpflow.transforms.Identity()
            self.assertTrue(self.m._needs_recompile)

    def test_replacement(self):
        with self.test_session():
            m = gpflow.model.Model()
            m.p = gpflow.param.Parameterized()
            m.p.p = gpflow.param.Param(1.0)
            m._needs_recompile = False
            # replace Parameterized
            new_p = gpflow.param.Parameterized()
            new_p.p = gpflow.param.Param(1.0)
            m.p = new_p
            self.assertTrue(m._needs_recompile is True)


class TestModelSessionGraphArguments(GPflowTestCase):
    """Tests for external graph and session passed to model."""

    class Dummy(gpflow.model.Model):
        """Dummy class with naive build_likelihood function"""
        def __init__(self):
            gpflow.model.Model.__init__(self)
            self.x = gpflow.param.Param(10)

        def build_likelihood(self):
            return tf.negative(tf.reduce_sum(tf.square(self.x)))

    def test_session_graph_properties(self):
        models = [TestModelSessionGraphArguments.Dummy()
                  for i in range(6)]
        m1, m2, m3, m4, m5, m6 = models
        session = tf.Session()
        graph = tf.Graph()
        m1.compile()
        m2.compile(session=session)
        m3.compile(graph=graph)

        with graph.as_default():
            m4.compile()

        m5.compile(session=session, graph=graph)
        with self.test_session() as sess_default:
            m6.compile()

        sessions = [m.session for m in models]
        sess1, sess2, sess3, sess4, sess5, sess6 = sessions
        sessions_set = set(map(str, sessions))
        self.assertNotEqual(sess_default, tf.get_default_graph())
        self.assertEqual(len(sessions_set), 5)
        self.assertEqual(sess2, sess5)
        self.assertEqual(sess1.graph, sess2.graph)
        self.assertEqual(sess3.graph, sess4.graph)
        self.assertEqual(sess2.graph, tf.get_default_graph())
        self.assertEqual(sess3.graph, graph)
        self.assertEqual(sess6.graph, sess_default.graph)
        self.assertEqual(sess6, sess_default)
        self.assertNotEqual(sess1.graph, sess3.graph)

        m6.compile(graph=sess_default.graph)
        self.assertEqual(sess6.graph, sess_default.graph)
        self.assertEqual(sess6, sess_default)

        [m.session.close() for m in models]


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


class TestKeyboardCatching(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            X = np.random.randn(1000, 3)
            Y = np.random.randn(1000, 3)
            Z = np.random.randn(100, 3)
            self.m = gpflow.sgpr.SGPR(X, Y, Z=Z, kern=gpflow.kernels.RBF(3))

    def test_optimize_np(self):
        with self.test_session():
            x0 = self.m.get_free_state()
            self.m.compile()
            self.m._objective = KeyboardRaiser(15, self.m._objective)
            self.m.optimize(disp=0, maxiter=1000, ftol=0, gtol=0)
            x1 = self.m.get_free_state()
            self.assertFalse(np.allclose(x0, x1))

    def test_optimize_tf(self):
        with self.test_session():
            x0 = self.m.get_free_state()
            callback = KeyboardRaiser(5, lambda x: None)
            o = tf.train.AdamOptimizer()
            self.m.optimize(o, maxiter=10, callback=callback)
            x1 = self.m.get_free_state()
            self.assertFalse(np.allclose(x0, x1))


class TestLikelihoodAutoflow(GPflowTestCase):
    def setUp(self):
        with self.test_session():
            X = np.random.randn(1000, 3)
            Y = np.random.randn(1000, 3)
            Z = np.random.randn(100, 3)
            self.m = gpflow.sgpr.SGPR(X, Y, Z=Z, kern=gpflow.kernels.RBF(3))

    def test_lik_and_prior(self):
        with self.test_session():
            l0 = self.m.compute_log_likelihood()
            p0 = self.m.compute_log_prior()
            self.m.kern.variance.prior = gpflow.priors.Gamma(1.4, 1.6)
            l1 = self.m.compute_log_likelihood()
            p1 = self.m.compute_log_prior()

            self.assertTrue(p0 == 0.0)
            self.assertFalse(p0 == p1)
            self.assertTrue(l0 == l1)


class TestName(GPflowTestCase):
    def test_name(self):
        m = gpflow.model.Model(name='foo')
        self.assertEqual(m.name, 'foo')


class TestNoRecompileThroughNewModelInstance(GPflowTestCase):
    """ Regression tests for Bug #454 """

    def setUp(self):
        self.X = np.random.rand(10, 2)
        self.Y = np.random.rand(10, 1)

    def test_gpr(self):
        with self.test_session():
            m1 = gpflow.gpr.GPR(self.X, self.Y, gpflow.kernels.Matern32(2))
            m1.compile()
            m2 = gpflow.gpr.GPR(self.X, self.Y, gpflow.kernels.Matern32(2))
            self.assertFalse(m1._needs_recompile)

    def test_sgpr(self):
        with self.test_session():
            m1 = gpflow.sgpr.SGPR(self.X, self.Y, gpflow.kernels.Matern32(2), Z=self.X)
            m1.compile()
            m2 = gpflow.sgpr.SGPR(self.X, self.Y, gpflow.kernels.Matern32(2), Z=self.X)
            self.assertFalse(m1._needs_recompile)

    def test_gpmc(self):
        with self.test_session():
            m1 = gpflow.gpmc.GPMC(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT())
            m1.compile()
            m2 = gpflow.gpmc.GPMC(
                    self.X, self.Y,
                    gpflow.kernels.Matern32(2),
                    likelihood=gpflow.likelihoods.StudentT())
            self.assertFalse(m1._needs_recompile)

    def test_sgpmc(self):
        with self.test_session():
            m1 = gpflow.sgpmc.SGPMC(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X)
            m1.compile()
            m2 = gpflow.sgpmc.SGPMC(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X)
            self.assertFalse(m1._needs_recompile)

    def test_svgp(self):
        with self.test_session():
            m1 = gpflow.svgp.SVGP(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X)
            m1.compile()
            m2 = gpflow.svgp.SVGP(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT(),
                Z=self.X)
            self.assertFalse(m1._needs_recompile)

    def test_vgp(self):
        with self.test_session():
            m1 = gpflow.vgp.VGP(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT())
            m1.compile()
            m2 = gpflow.vgp.VGP(
                self.X, self.Y,
                gpflow.kernels.Matern32(2),
                likelihood=gpflow.likelihoods.StudentT())
            self.assertFalse(m1._needs_recompile)


if __name__ == "__main__":
    unittest.main()
