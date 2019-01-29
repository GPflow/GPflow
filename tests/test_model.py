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
# limitations under the License.

import tensorflow as tf

import numpy as np
from numpy.testing import assert_almost_equal

import gpflow
from gpflow.test_util import GPflowTestCase


class Quadratic(gpflow.models.Model):
    def __init__(self):
        rng = np.random.RandomState(0)
        gpflow.models.Model.__init__(self)
        self.x = gpflow.Param(rng.randn(10))

    @gpflow.params_as_tensors
    def _build_likelihood(self):
        return tf.negative(tf.reduce_sum(tf.square(self.x)))


class TestOptimize(GPflowTestCase):
    def test_adam(self):
        with self.test_context():
            m = Quadratic()
            opt = gpflow.train.AdamOptimizer(0.01)
            opt.minimize(m, maxiter=5000)
            self.assertTrue(m.x.read_value().max() < 1e-2)

    def test_lbfgsb(self):
        with self.test_context():
            m = Quadratic()
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(m, maxiter=1000)
            self.assertTrue(m.x.read_value().max() < 1e-6)


class Empty(gpflow.models.Model):
    def __init__(self, *args, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'Empty'
        super().__init__(*args, **kwargs)
    def _build_likelihood(self):
        return tf.convert_to_tensor(1., dtype=gpflow.settings.float_type)


class EmptyTest(GPflowTestCase):
    def test_compile_model_without_parameters(self):
        with self.test_context():
            m = Empty()
            assert_almost_equal(m.compute_log_likelihood(), 1.0)
            assert_almost_equal(m.compute_log_prior(), 0.0)

    def test_parameters_list_empty(self):
        with self.test_context():
            m = Empty(autobuild=False)
            self.assertEqual(list(m.parameters), [])
            self.assertEqual(list(m.trainable_parameters), [])
            self.assertEqual(list(m.params), [])
            m.compile()
            self.assertEqual(list(m.parameters), [])
            self.assertEqual(list(m.trainable_parameters), [])
            self.assertEqual(list(m.params), [])

    def test_objective_tensor(self):
        with self.test_context():
            m = Empty(autobuild=False)
            self.assertEqual(m.objective, None)
            m.build()
            self.assertTrue(gpflow.misc.is_tensor(m.objective))


class ReplaceParameterTest(GPflowTestCase):

    class Origin(gpflow.models.Model):
        def __init__(self):
            super(ReplaceParameterTest.Origin, self).__init__()
            self.a = gpflow.Param(1.)
            self.b = gpflow.Param(2.)

        @gpflow.params_as_tensors
        def _build_likelihood(self):
            return tf.square(self.a) + tf.square(self.b)

    def test_replace_parameter(self):
        class OriginSuccess(ReplaceParameterTest.Origin):
            def __init__(self):
                super(OriginSuccess, self).__init__()
                self.b = gpflow.Param(np.array(3.))

        class OriginAllDataholders(ReplaceParameterTest.Origin):
            def __init__(self):
                super(OriginAllDataholders, self).__init__()
                self.a = gpflow.DataHolder(np.array(2.))
                self.b = gpflow.DataHolder(np.array(2.))

        with self.test_context():
            m0 = self.Origin()
            m0.compile()
            assert_almost_equal(m0.compute_log_likelihood(), 5.0)

            m1 = OriginSuccess()
            m1.compile()
            assert_almost_equal(m1.compute_log_likelihood(), 10.0)

            m2 = OriginAllDataholders()
            m2.compile()
            assert_almost_equal(m2.compute_log_likelihood(), 8.0)


class KeyboardRaiser:
    """
    This wraps a function and makes it raise a KeyboardInterrupt after some number of calls
    """

    def __init__(self, iters_to_raise):
        self.iters_to_raise = iters_to_raise
        self.count = 0

    def __call__(self, *a, **kw):
        self.count += 1
        if self.count >= self.iters_to_raise:
            raise KeyboardInterrupt

def setup_sgpr():
    X = np.random.randn(1000, 3)
    Y = np.random.randn(1000, 3)
    Z = np.random.randn(100, 3)
    return gpflow.models.SGPR(X, Y, Z=Z, kern=gpflow.kernels.RBF(3))


class TestLikelihoodAutoflow(GPflowTestCase):
    def test_lik_and_prior(self):
        with self.test_context(graph=tf.Graph()):
            m = setup_sgpr()
            l0 = m.compute_log_likelihood()
            p0 = m.compute_log_prior()

        m.clear()

        with self.test_context(graph=tf.Graph()):
            m.kern.variance.prior = gpflow.priors.Gamma(1.4, 1.6)
            m.compile()
            l1 = m.compute_log_likelihood()
            p1 = m.compute_log_prior()

        self.assertEqual(p0, 0.0)
        self.assertNotEqual(p0, p1)
        self.assertEqual(l0, l1)


class TestName(GPflowTestCase):
    def test_name(self):
        with self.test_context():
            m1 = Empty()
            self.assertEqual(m1.name, 'Empty')
            m2 = Empty(name='foo')
            self.assertEqual(m2.name, 'foo')


class EvalDataSVGP(gpflow.models.SVGP):
    @gpflow.decors.autoflow()
    @gpflow.decors.params_as_tensors
    def XY(self):
        return self.X, self.Y


class TestMinibatchSVGP(GPflowTestCase):
    def test_minibatch_sync(self):
        with self.test_context():
            X = np.random.randn(1000, 1)
            Y = X.copy()
            Z = X[:100, :].copy()
            size = 10
            m = EvalDataSVGP(X, Y, gpflow.kernels.RBF(1),
                             gpflow.likelihoods.Gaussian(),
                             minibatch_size=size, Z=Z)

            eX_prev, eY_prev = np.random.randn(size, 1), np.random.randn(size, 1)
            for _ in range(10):
                eX, eY = m.XY()
                assert not np.allclose(eX, eX_prev)
                assert not np.allclose(eY, eY_prev)
                assert np.allclose(eX, eY)
                eX_prev, eY_prev = eX, eY

# class TestNoRecompileThroughNewModelInstance(GPflowTestCase):
#     """ Regression tests for Bug #454 """

#     def setUp(self):
#         self.X = np.random.rand(10, 2)
#         self.Y = np.random.rand(10, 1)

#     def test_gpr(self):
#         with self.test_context():
#             m1 = gpflow.models.GPR(self.X, self.Y, gpflow.kernels.Matern32(2))
#             m1.compile()
#             m2 = gpflow.models.GPR(self.X, self.Y, gpflow.kernels.Matern32(2))
#             self.assertFalse(m1._needs_recompile)

#     def test_sgpr(self):
#         with self.test_context():
#             m1 = gpflow.models.SGPR(self.X, self.Y, gpflow.kernels.Matern32(2), Z=self.X)
#             m1.compile()
#             m2 = gpflow.models.SGPR(self.X, self.Y, gpflow.kernels.Matern32(2), Z=self.X)
#             self.assertFalse(m1._needs_recompile)

#     def test_gpmc(self):
#         with self.test_context():
#             m1 = gpflow.models.GPMC(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT())
#             m1.compile()
#             m2 = gpflow.models.GPMC(
#                     self.X, self.Y,
#                     gpflow.kernels.Matern32(2),
#                     likelihood=gpflow.likelihoods.StudentT())
#             self.assertFalse(m1._needs_recompile)

#     def test_sgpmc(self):
#         with self.test_context():
#             m1 = gpflow.models.SGPMC(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT(),
#                 Z=self.X)
#             m1.compile()
#             m2 = gpflow.models.SGPMC(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT(),
#                 Z=self.X)
#             self.assertFalse(m1._needs_recompile)

#     def test_svgp(self):
#         with self.test_context():
#             m1 = gpflow.models.SVGP(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT(),
#                 Z=self.X)
#             m1.compile()
#             m2 = gpflow.models.SVGP(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT(),
#                 Z=self.X)
#             self.assertFalse(m1._needs_recompile)

#     def test_vgp(self):
#         with self.test_context():
#             m1 = gpflow.models.VGP(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT())
#             m1.compile()
#             m2 = gpflow.models.VGP(
#                 self.X, self.Y,
#                 gpflow.kernels.Matern32(2),
#                 likelihood=gpflow.likelihoods.StudentT())
#             self.assertFalse(m1._needs_recompile)


if __name__ == "__main__":
    tf.test.main()
