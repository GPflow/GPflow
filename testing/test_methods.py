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

import GPflow
import numpy as np
import unittest
import tensorflow as tf


class TestMethods(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100, 2)
        self.Y = self.rng.randn(100, 1)
        self.Z = self.rng.randn(10, 2)
        self.lik = GPflow.likelihoods.Gaussian()
        self.kern = GPflow.kernels.Matern32(2)
        self.Xs = self.rng.randn(10, 2)

        # make one of each model
        self.ms = []
        #for M in (GPflow.gpmc.GPMC, GPflow.vgp.VGP):
        for M in (GPflow.vgp.VGP, GPflow.gpmc.GPMC):
            self.ms.append(M(self.X, self.Y, self.kern, self.lik))
        for M in (GPflow.sgpmc.SGPMC, GPflow.svgp.SVGP):
            self.ms.append(M(self.X, self.Y, self.kern, self.lik, self.Z))
        self.ms.append(GPflow.gpr.GPR(self.X, self.Y, self.kern))
        self.ms.append(GPflow.sgpr.SGPR(self.X, self.Y, self.kern, Z=self.Z))
        self.ms.append(GPflow.sgpr.GPRFITC(self.X, self.Y, self.kern, Z=self.Z))

    def test_all(self):
        # test sizes.
        for m in self.ms:
            m._compile()
            f, g = m._objective(m.get_free_state())
            self.assertTrue(f.size == 1)
            self.assertTrue(g.size == m.get_free_state().size)

    def test_tf_optimize(self):
        for m in self.ms:
            trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            if isinstance(m, (GPflow.gpr.GPR,GPflow.vgp.VGP,GPflow.svgp.SVGP)):
                optimizeOp = m._compile(trainer)
                self.assertTrue(optimizeOp is not None)


    def test_predict_f(self):
        for m in self.ms:
            mf, vf = m.predict_f(self.Xs)
            self.assertTrue(mf.shape == vf.shape)
            self.assertTrue(mf.shape == (10, 1))
            self.assertTrue(np.all(vf >= 0.0))

    def test_predict_y(self):
        for m in self.ms:
            mf, vf = m.predict_y(self.Xs)
            self.assertTrue(mf.shape == vf.shape)
            self.assertTrue(mf.shape == (10, 1))
            self.assertTrue(np.all(vf >= 0.0))

    def test_predict_density(self):
        self.Ys = self.rng.randn(10, 1)
        for m in self.ms:
            d = m.predict_density(self.Xs, self.Ys)
            self.assertTrue(d.shape == (10, 1))

    def test_predict_f_gradients(self):
        for m in self.ms:
            mf, vf = m.predict_f_gradients(self.Xs)
            self.assertTrue(mf.shape == vf.shape)
            self.assertTrue(mf.shape == (10, 2))

    def test_predict_y_gradients(self):
        for m in self.ms:
            mf, vf = m.predict_y_gradients(self.Xs)
            self.assertTrue(mf.shape == vf.shape)
            self.assertTrue(mf.shape == (10, 2))

class TestSVGP(unittest.TestCase):
    """
    The SVGP has four modes of operation. with and without whitening, with and
    without diagonals.

    Here we make sure thet the bound on the likelihood is the same when using
    both representations (as far as possible)
    """
    def setUp(self):
        tf.reset_default_graph()
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(20, 1)
        self.Y = self.rng.randn(20, 2)
        self.Z = self.rng.randn(3, 1)

    def test_white(self):
        m1 = GPflow.svgp.SVGP(self.X, self.Y,
                              kern=GPflow.kernels.RBF(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z, q_diag=True, whiten=True)
        m2 = GPflow.svgp.SVGP(self.X, self.Y,
                              kern=GPflow.kernels.RBF(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z, q_diag=False, whiten=True)
        m1._compile()
        m2._compile()

        qsqrt, qmean = self.rng.randn(2, 3, 2)
        qsqrt = (qsqrt**2)*0.01
        m1.q_sqrt = qsqrt
        m1.q_mu = qmean
        m2.q_sqrt = np.array([np.diag(qsqrt[:, 0]),
                              np.diag(qsqrt[:, 1])]).swapaxes(0, 2)
        m2.q_mu = qmean
        self.assertTrue(np.allclose(m1._objective(m1.get_free_state())[0],
                                    m2._objective(m2.get_free_state())[0]))

    def test_notwhite(self):
        m1 = GPflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=GPflow.kernels.RBF(1) +
                                   GPflow.kernels.White(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=True,
                              whiten=False)
        m2 = GPflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=GPflow.kernels.RBF(1) +
                                   GPflow.kernels.White(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=False,
                              whiten=False)
        m1._compile()
        m2._compile()

        qsqrt, qmean = self.rng.randn(2, 3, 2)
        qsqrt = (qsqrt**2)*0.01
        m1.q_sqrt = qsqrt
        m1.q_mu = qmean
        m2.q_sqrt = np.array([np.diag(qsqrt[:, 0]), np.diag(qsqrt[:, 1])]).swapaxes(0, 2)
        m2.q_mu = qmean
        self.assertTrue(np.allclose(m1._objective(m1.get_free_state())[0],
                                    m2._objective(m2.get_free_state())[0]))

    def test_q_sqrt_fixing(self):
        """
        In response to bug #46, we need to make sure that the q_sqrt matrix can be fixed
        """
        m1 = GPflow.svgp.SVGP(self.X, self.Y,
                              kern=GPflow.kernels.RBF(1) + GPflow.kernels.White(1),
                              likelihood=GPflow.likelihoods.Exponential(),
                              Z=self.Z)
        m1.q_sqrt.fixed = True
        m1._compile()


class TestSparseMCMC(unittest.TestCase):
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    def setUp(self):
        tf.reset_default_graph()
        rng = np.random.RandomState(0)
        X = rng.randn(10, 1)
        Y = rng.randn(10, 1)
        v_vals = rng.randn(10, 1)

        lik = GPflow.likelihoods.StudentT
        self.m1 = GPflow.gpmc.GPMC(X=X, Y=Y, kern=GPflow.kernels.Exponential(1), likelihood=lik())
        self.m2 = GPflow.sgpmc.SGPMC(X=X, Y=Y, kern=GPflow.kernels.Exponential(1), likelihood=lik(), Z=X.copy())

        self.m1.V = v_vals
        self.m2.V = v_vals.copy()
        self.m1.kern.lengthscale = .8
        self.m2.kern.lengthscale = .8
        self.m1.kern.variance = 4.2
        self.m2.kern.variance = 4.2

        self.m1._compile()
        self.m2._compile()

    def test_likelihoods_and_gradients(self):
        f1, _ = self.m1._objective(self.m1.get_free_state())
        f2, _ = self.m2._objective(self.m2.get_free_state())
        self.assertTrue(np.allclose(f1, f2))
        # the parameters might not be in the same order, so
        # sort the gradients before checking they're the same
        _, g1 = self.m1._objective(self.m1.get_free_state())
        _, g2 = self.m2._objective(self.m2.get_free_state())
        g1 = np.sort(g1)
        g2 = np.sort(g2)
        self.assertTrue(np.allclose(g1, g2, 1e-4))


if __name__ == "__main__":
    unittest.main()
