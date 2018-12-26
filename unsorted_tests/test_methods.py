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
from numpy.testing import assert_array_equal, assert_array_less, assert_allclose

import gpflow
from gpflow.test_util import GPflowTestCase


class TestMethods(GPflowTestCase):
    def prepare(self):
        rng = np.random.RandomState(0)
        X = rng.randn(100, 2)
        Y = rng.randn(100, 1)
        Z = rng.randn(10, 2)
        lik = gpflow.likelihoods.Gaussian()
        kern = gpflow.kernels.Matern32(2)
        Xs = rng.randn(10, 2)

        # make one of each model
        ms = []
        #for M in (gpflow.models.GPMC, gpflow.models.VGP):
        for M in (gpflow.models.VGP, gpflow.models.GPMC):
            ms.append(M(X, Y, kern, lik))
        for M in (gpflow.models.SGPMC, gpflow.models.SVGP):
            ms.append(M(X, Y, kern, lik, Z))
        ms.append(gpflow.models.GPR(X, Y, kern))
        ms.append(gpflow.models.SGPR(X, Y, kern, Z=Z))
        ms.append(gpflow.models.GPRFITC(X, Y, kern, Z=Z))
        return ms, Xs, rng

    def test_all(self):
        # test sizes.
        with self.test_context():
            ms, _Xs, _rng = self.prepare()
            for m in ms:
                self.assertEqual(m.is_built_coherence(), gpflow.Build.YES)

    def test_predict_f(self):
        with self.test_context():
            ms, Xs, _rng = self.prepare()
            for m in ms:
                mf, vf = m.predict_f(Xs)
                assert_array_equal(mf.shape, vf.shape)
                assert_array_equal(mf.shape, (10, 1))
                assert_array_less(np.full_like(vf, -1e-6), vf)

    def test_predict_y(self):
        with self.test_context():
            ms, Xs, _rng = self.prepare()
            for m in ms:
                mf, vf = m.predict_y(Xs)
                assert_array_equal(mf.shape, vf.shape)
                assert_array_equal(mf.shape, (10, 1))
                assert_array_less(np.full_like(vf, -1e-6), vf)

    def test_predict_density(self):
        with self.test_context():
            ms, Xs, rng = self.prepare()
            Ys = rng.randn(10, 1)
            for m in ms:
                d = m.predict_density(Xs, Ys)
                assert_array_equal(d.shape, (10, 1))


class TestSVGP(GPflowTestCase):
    """
    The SVGP has four modes of operation. with and without whitening, with and
    without diagonals.

    Here we make sure that the bound on the likelihood is the same when using
    both representations (as far as possible)
    """
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(20, 1)
        self.Y = self.rng.randn(20, 2)**2
        self.Z = self.rng.randn(3, 1)

    def test_white(self):
        with self.test_context() as session:
            m1 = gpflow.models.SVGP(
                self.X, self.Y,
                kern=gpflow.kernels.RBF(1),
                likelihood=gpflow.likelihoods.Exponential(),
                Z=self.Z,
                q_diag=True,
                whiten=True)
            m2 = gpflow.models.SVGP(
                self.X, self.Y,
                kern=gpflow.kernels.RBF(1),
                likelihood=gpflow.likelihoods.Exponential(),
                Z=self.Z,
                q_diag=False,
                whiten=True)
            qsqrt, qmean = self.rng.randn(2, 3, 2)
            qsqrt = (qsqrt**2) * 0.01
            m1.q_sqrt = qsqrt
            m1.q_mu = qmean
            m2.q_sqrt = np.array([np.diag(qsqrt[:, 0]), np.diag(qsqrt[:, 1])])
            m2.q_mu = qmean

            obj1 = session.run(m1.objective, feed_dict=m1.feeds)
            obj2 = session.run(m2.objective, feed_dict=m2.feeds)
            assert_allclose(obj1, obj2)

    def test_notwhite(self):
        with self.test_context() as session:
            m1 = gpflow.models.SVGP(
                self.X,
                self.Y,
                kern=gpflow.kernels.RBF(1) + gpflow.kernels.White(1),
                likelihood=gpflow.likelihoods.Exponential(),
                Z=self.Z,
                q_diag=True,
                whiten=False)
            m2 = gpflow.models.SVGP(
                self.X,
                self.Y,
                kern=gpflow.kernels.RBF(1) + gpflow.kernels.White(1),
                likelihood=gpflow.likelihoods.Exponential(),
                Z=self.Z,
                q_diag=False,
                whiten=False)
            qsqrt, qmean = self.rng.randn(2, 3, 2)
            qsqrt = (qsqrt**2)*0.01
            m1.q_sqrt = qsqrt
            m1.q_mu = qmean
            m2.q_sqrt = np.array([np.diag(qsqrt[:, 0]), np.diag(qsqrt[:, 1])])
            m2.q_mu = qmean
            obj1 = session.run(m1.objective, feed_dict=m1.feeds)
            obj2 = session.run(m2.objective, feed_dict=m2.feeds)
            assert_allclose(obj1, obj2)

    def test_q_sqrt_fixing(self):
        """
        In response to bug #46, we need to make sure that the q_sqrt matrix can be fixed
        """
        with self.test_context() as session:
            m1 = gpflow.models.SVGP(
                self.X, self.Y,
                kern=gpflow.kernels.RBF(1) + gpflow.kernels.White(1),
                likelihood=gpflow.likelihoods.Exponential(),
                Z=self.Z)
            m1.q_sqrt.trainable = False

class TestStochasticGradients(GPflowTestCase):
    """
    In response to bug #281, we need to make sure stochastic update
    happens correctly in tf optimizer mode.
    To do this compare stochastic updates with deterministic updates
    that should be equivalent.

    Data term in svgp likelihood is
    \sum_{i=1^N}E_{q(i)}[\log p(y_i | f_i )

    This sum is then approximated with an unbiased minibatch estimate.
    In this test we substitute a deterministic analogue of the batchs
    sampler for which we can predict the effects of different updates.
    """
    def setUp(self):
        tf.set_random_seed(0)
        self.XAB = np.atleast_2d(np.array([0., 1.])).T
        self.YAB = np.atleast_2d(np.array([-1., 3.])).T
        self.sharedZ = np.atleast_2d(np.array([0.5]) )
        self.indexA = 0
        self.indexB = 1

    def get_indexed_data(self, baseX, baseY, indices):
        newX = baseX[indices]
        newY = baseY[indices]
        return newX, newY

    def get_model(self, X, Y, Z, minibatch_size):
        model = gpflow.models.SVGP(
            X, Y, kern=gpflow.kernels.RBF(1),
            likelihood=gpflow.likelihoods.Gaussian(),
            Z=Z, minibatch_size=minibatch_size)
        return model

    def get_opt(self):
        learning_rate = .001
        opt = gpflow.train.GradientDescentOptimizer(learning_rate, use_locking=True)
        return opt

    def get_indexed_model(self, X, Y, Z, minibatch_size, indices):
        Xindices, Yindices = self.get_indexed_data(X, Y, indices)
        indexedModel = self.get_model(Xindices, Yindices, Z, minibatch_size)
        return indexedModel

    def check_models_close(self, m1, m2, tolerance=1e-2):
        m1_params = {p.pathname: p for p in list(m1.trainable_parameters)}
        m2_params = {p.pathname: p for p in list(m2.trainable_parameters)}
        if set(m2_params.keys()) != set(m2_params.keys()):
            return False
        for key in m1_params:
            p1 = m1_params[key]
            p2 = m2_params[key]
            if not np.allclose(p1.read_value(), p2.read_value(), rtol=tolerance, atol=tolerance):
                return False
        return True

    def compare_models(self, indicesOne, indicesTwo,
                         batchOne, batchTwo, maxiter, checkSame=True):
        m1 = self.get_indexed_model(self.XAB, self.YAB, self.sharedZ, batchOne, indicesOne)
        m2 = self.get_indexed_model(self.XAB, self.YAB, self.sharedZ, batchTwo, indicesTwo)

        opt1 = self.get_opt()
        opt2 = self.get_opt()

        opt1.minimize(m1, maxiter=maxiter)
        opt2.minimize(m2, maxiter=maxiter)
        if checkSame:
            self.assertTrue(self.check_models_close(m1, m2))
        else:
            self.assertFalse(self.check_models_close(m1, m2))

    # TODO(@awav):
    # These three tests below can be extremly unstable on different machines
    # and different settings.

    def testOne(self):
        with self.test_context():
            self.compare_models(
                [self.indexA, self.indexB],
                [self.indexB, self.indexA],
                batchOne=2, batchTwo=2, maxiter=3)

    def testTwo(self):
        with self.test_context():
            self.compare_models(
                [self.indexA, self.indexB],
                [self.indexA, self.indexA],
                batchOne=1, batchTwo=2, maxiter=1)

    def testThree(self):
        with self.test_context():
            self.compare_models(
                [self.indexA, self.indexA],
                [self.indexA, self.indexB],
                batchOne=1, batchTwo=1, maxiter=2)

class TestSparseMCMC(GPflowTestCase):
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    def test_likelihoods_and_gradients(self):
        with self.test_context() as session:
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            Y = rng.randn(10, 1)
            v_vals = rng.randn(10, 1)

            lik = gpflow.likelihoods.StudentT

            m1 = gpflow.models.GPMC(
                X=X, Y=Y,
                kern=gpflow.kernels.Exponential(1),
                likelihood=lik())

            m2 = gpflow.models.SGPMC(
                X=X, Y=Y,
                kern=gpflow.kernels.Exponential(1),
                likelihood=lik(), Z=X.copy())

            m1.V = v_vals
            m2.V = v_vals.copy()
            m1.kern.lengthscale = .8
            m2.kern.lengthscale = .8
            m1.kern.variance = 4.2
            m2.kern.variance = 4.2

            f1 = session.run(m1.objective)
            f2 = session.run(m2.objective)
            assert_allclose(f1, f2)

            # the parameters might not be in the same order, so
            # sort the gradients before checking they're the same
            # g1 = self.m1.objective(self.m1.get_free_state())
            # g2 = self.m2.objective(self.m2.get_free_state())
            # g1 = np.sort(g1)
            # g2 = np.sort(g2)
            # self.assertTrue(np.allclose(g1, g2, 1e-4))


if __name__ == "__main__":
    tf.test.main()
