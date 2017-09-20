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

import gpflow
from gpflow.minibatch import SequenceIndices
import numpy as np
import unittest
import tensorflow as tf
import gpflow

from testing.gpflow_testcase import GPflowTestCase
from gpflow.minibatch import SequenceIndices

class TestMethods(GPflowTestCase):
    def setUp(self):
        self.rng = np.random.RandomState(0)
        self.X = self.rng.randn(100, 2)
        self.Y = self.rng.randn(100, 1)
        self.Z = self.rng.randn(10, 2)
        self.lik = gpflow.likelihoods.Gaussian()
        self.kern = gpflow.kernels.Matern32(2)
        self.Xs = self.rng.randn(10, 2)

        # make one of each model
        self.ms = []
        #for M in (gpflow.gpmc.GPMC, gpflow.vgp.VGP):
        for M in (gpflow.vgp.VGP, gpflow.gpmc.GPMC):
            self.ms.append(M(self.X, self.Y, self.kern, self.lik))
        for M in (gpflow.sgpmc.SGPMC, gpflow.svgp.SVGP):
            self.ms.append(M(self.X, self.Y, self.kern, self.lik, self.Z))
        self.ms.append(gpflow.gpr.GPR(self.X, self.Y, self.kern))
        self.ms.append(gpflow.sgpr.SGPR(self.X, self.Y, self.kern, Z=self.Z))
        self.ms.append(gpflow.sgpr.GPRFITC(self.X, self.Y, self.kern, Z=self.Z))

    def test_all(self):
        # test sizes.
        for m in self.ms:
            m.compile()
            f, g = m._objective(m.get_free_state())
            self.assertTrue(f.size == 1)
            self.assertTrue(g.size == m.get_free_state().size)

    def test_tf_optimize(self):
        for m in self.ms:
            trainer = tf.train.AdamOptimizer(learning_rate=0.001)
            if isinstance(m, (gpflow.gpr.GPR, gpflow.vgp.VGP,
                              gpflow.svgp.SVGP, gpflow.gpmc.GPMC)):
                optimizeOp = m.compile(optimizer=trainer)
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
        m1 = gpflow.svgp.SVGP(self.X, self.Y,
                              kern=gpflow.kernels.RBF(1),
                              likelihood=gpflow.likelihoods.Exponential(),
                              Z=self.Z, q_diag=True, whiten=True)
        m2 = gpflow.svgp.SVGP(self.X, self.Y,
                              kern=gpflow.kernels.RBF(1),
                              likelihood=gpflow.likelihoods.Exponential(),
                              Z=self.Z, q_diag=False, whiten=True)
        m1.compile()
        m2.compile()

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
        m1 = gpflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=gpflow.kernels.RBF(1) +
                                   gpflow.kernels.White(1),
                              likelihood=gpflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=True,
                              whiten=False)
        m2 = gpflow.svgp.SVGP(self.X,
                              self.Y,
                              kern=gpflow.kernels.RBF(1) +
                                   gpflow.kernels.White(1),
                              likelihood=gpflow.likelihoods.Exponential(),
                              Z=self.Z,
                              q_diag=False,
                              whiten=False)
        m1.compile()
        m2.compile()

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
        m1 = gpflow.svgp.SVGP(self.X, self.Y,
                              kern=gpflow.kernels.RBF(1) + gpflow.kernels.White(1),
                              likelihood=gpflow.likelihoods.Exponential(),
                              Z=self.Z)
        m1.q_sqrt.fixed = True
        m1.compile()

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
        self.XAB = np.atleast_2d(np.array([0.,1.])).T
        self.YAB = np.atleast_2d(np.array([-1.,3.])).T
        self.sharedZ = np.atleast_2d(np.array([0.5]) )
        self.indexA = 0
        self.indexB = 1

    def getIndexedData(self,baseX,baseY,indeces):
        newX = baseX[indeces]
        newY = baseY[indeces]
        return newX, newY

    def getModel(self,X,Y,Z,minibatch_size):
        model = gpflow.svgp.SVGP(X,
                                 Y,
                                 kern = gpflow.kernels.RBF(1),
                                 likelihood = gpflow.likelihoods.Gaussian(),
                                 Z = Z,
                                 minibatch_size=minibatch_size)
        #This step changes the batch indeces to cycle.
        model.X.index_manager = SequenceIndices(minibatch_size,X.shape[0])
        model.Y.index_manager = SequenceIndices(minibatch_size,X.shape[0])
        return model

    def getTfOptimizer(self):
        learning_rate = .1
        opt = tf.train.GradientDescentOptimizer(learning_rate,
                                                use_locking=True)
        return opt

    def getIndexedModel(self,X,Y,Z,minibatch_size,indeces):
        Xindeces,Yindeces = self.getIndexedData(X,Y,indeces)
        indexedModel = self.getModel(Xindeces,Yindeces,Z,minibatch_size)
        return indexedModel

    def checkModelsClose(self,modelA,modelB,tolerance=1e-2):
        modelA_dict = modelA.get_parameter_dict()
        modelB_dict = modelB.get_parameter_dict()
        if sorted(modelA_dict.keys())!=sorted(modelB_dict.keys()):
            return False
        for key in modelA_dict:
            if ((modelA_dict[key] - modelB_dict[key])>tolerance).any():
                return False
        return True

    def compareTwoModels(self,indecesOne,indecesTwo,
                              batchOne,batchTwo,
                              maxiter,
                              checkSame=True):
        modelOne = self.getIndexedModel(self.XAB,
                                        self.YAB,
                                        self.sharedZ,
                                        batchOne,
                                        indecesOne)
        modelTwo = self.getIndexedModel(self.XAB,
                                        self.YAB,
                                        self.sharedZ,
                                        batchTwo,
                                        indecesTwo)
        modelOne.optimize(method=self.getTfOptimizer(),maxiter=maxiter)
        modelTwo.optimize(method=self.getTfOptimizer(),maxiter=maxiter)
        if checkSame:
            self.assertTrue(self.checkModelsClose(modelOne,modelTwo))
        else:
            self.assertFalse(self.checkModelsClose(modelOne,modelTwo))

    def testOne(self):
        self.compareTwoModels([self.indexA,self.indexB],
                              [self.indexB,self.indexA],
                              2,
                              2,
                              3)

    def testTwo(self):
        self.compareTwoModels([self.indexA,self.indexB],
                              [self.indexA,self.indexA],
                              1,
                              2,
                              1)

    def testThree(self):
        self.compareTwoModels([self.indexA,self.indexA],
                              [self.indexA,self.indexB],
                              1,
                              1,
                              2,
                              False)

class TestSparseMCMC(GPflowTestCase):
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    def setUp(self):
        with self.test_session():
            rng = np.random.RandomState(0)
            X = rng.randn(10, 1)
            Y = rng.randn(10, 1)
            v_vals = rng.randn(10, 1)

            lik = gpflow.likelihoods.StudentT
            self.m1 = gpflow.gpmc.GPMC(
                X=X, Y=Y, kern=gpflow.kernels.Exponential(1), likelihood=lik())
            self.m2 = gpflow.sgpmc.SGPMC(
                X=X, Y=Y,
                kern=gpflow.kernels.Exponential(1),
                likelihood=lik(), Z=X.copy())

            self.m1.V = v_vals
            self.m2.V = v_vals.copy()
            self.m1.kern.lengthscale = .8
            self.m2.kern.lengthscale = .8
            self.m1.kern.variance = 4.2
            self.m2.kern.variance = 4.2

            self.m1.compile()
            self.m2.compile()

    def test_likelihoods_and_gradients(self):
        with self.test_session():
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
