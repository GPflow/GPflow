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
import tensorflow as tf
import numpy as np
import unittest
from .reference import referenceRbfKernel

from testing.gpflow_testcase import GPflowTestCase


def referenceUnivariateLogMarginalLikelihood(y, K, noiseVariance):
    return (-0.5 * y * y / (K + noiseVariance)
            -0.5 * np.log(K + noiseVariance)
            -0.5 * np.log(np.pi * 2.))


def referenceUnivariatePosterior(y, K, noiseVariance):
    mean = K * y / (K + noiseVariance)
    variance = K - K / (K + noiseVariance)
    return mean, variance


def referenceUnivariatePriorKL(meanA, meanB, varA, varB):
    # KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and qB are univariate normal distributions.
    return (0.5 * (np.log(varB) - np.log(varA) - 1. + varA/varB +
                   (meanB-meanA) * (meanB - meanA) / varB))


def referenceMultivariatePriorKL(meanA, covA, meanB, covB):
    # KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and aB are
    # K dimensional multivariate normal distributions.
    # Analytically tractable and equal to...
    # 0.5 * (Tr(covB^{-1} covA) + (meanB - meanA)^T covB^{-1} (meanB - meanA)
    #        - K + log(det(covB)) - log (det(covA)))
    K = covA.shape[0]
    traceTerm = 0.5 * np.trace(np.linalg.solve(covB, covA))
    delta = meanB - meanA
    mahalanobisTerm = 0.5 * np.dot(delta.T, np.linalg.solve(covB, delta))
    constantTerm = -0.5 * K
    priorLogDeterminantTerm = 0.5*np.linalg.slogdet(covB)[1]
    variationalLogDeterminantTerm = -0.5 * np.linalg.slogdet(covA)[1]
    return traceTerm + mahalanobisTerm + constantTerm + priorLogDeterminantTerm + variationalLogDeterminantTerm


def kernel(kernelVariance=1, lengthScale=1.):
    kern = gpflow.kernels.RBF(1)
    kern.variance = kernelVariance
    kern.lengthscales = lengthScale
    return kern


class VariationalUnivariateTest(GPflowTestCase):
    def setUp(self):
        self.y_real = 2.
        self.K = 1.
        self.noiseVariance = 0.5
        self.univariate = 1
        self.oneLatentFunction = 1
        self.meanZero = 0.
        self.X = np.atleast_2d(np.array([0.]))
        self.Y = np.atleast_2d(np.array([self.y_real]))
        self.Z = self.X.copy()
        self.lik = gpflow.likelihoods.Gaussian()
        self.lik.variance = self.noiseVariance
        self.posteriorMean, self.posteriorVariance = referenceUnivariatePosterior(
            y=self.y_real, K=self.K,
            noiseVariance=self.noiseVariance)
        self.posteriorStd = np.sqrt(self.posteriorVariance)

    def get_model(self, is_diagonal, is_whitened):
        m = gpflow.svgp.SVGP(X=self.X, Y=self.Y,
                             kern=kernel(kernelVariance=self.K),
                             likelihood=self.lik, Z=self.Z, q_diag=is_diagonal, whiten=is_whitened)
        if is_diagonal:
            m.q_sqrt = (np.ones((self.univariate, self.oneLatentFunction))
                        * self.posteriorStd)
        else:
            m.q_sqrt = (np.ones((self.univariate, self.univariate, self.oneLatentFunction))
                        * self.posteriorStd)
        m.q_mu = np.ones((self.univariate, self.oneLatentFunction)) * self.posteriorMean
        return m

    def test_prior_KL(self):
        with self.test_session():
            meanA = self.posteriorMean
            varA = self.posteriorVariance
            meanB = self.meanZero  # Assumes a zero
            varB = self.K

            referenceKL = referenceUnivariatePriorKL(meanA, meanB, varA, varB)

            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    m = self.get_model(is_diagonal, is_whitened)

                    test_prior_KL = gpflow.param.AutoFlow()(m.build_prior_KL.__func__)(m)
                    self.assertTrue(np.abs(referenceKL - test_prior_KL) < 1e-4)

    def test_build_likelihood(self):
        with self.test_session():
            # reference marginal likelihood
            log_marginal_likelihood = referenceUnivariateLogMarginalLikelihood(
                y=self.y_real, K=self.K, noiseVariance=self.noiseVariance)

            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    model = self.get_model(is_diagonal, is_whitened)
                    model_likelihood = model.compute_log_likelihood()
                    self.assertTrue(
                        np.abs(model_likelihood - log_marginal_likelihood) < 1e-4)

    def testUnivariateConditionals(self):
        with self.test_session() as sess:
            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    m = self.get_model(is_diagonal, is_whitened)
                    free_vars = tf.placeholder(tf.float64)
                    m.make_tf_array(free_vars)
                    with m.tf_mode():
                        if is_whitened:
                            args = (self.X,
                                    self.Z,
                                    m.kern,
                                    m.q_mu,
                                    m.q_sqrt,
                                    self.oneLatentFunction)
                            fmean_func, fvar_func = gpflow.conditionals.gaussian_gp_predict_whitened(*args)
                        else:
                            args = (self.X,
                                    self.Z,
                                    m.kern,
                                    m.q_mu,
                                    m.q_sqrt,
                                    self.oneLatentFunction)
                            fmean_func, fvar_func = gpflow.conditionals.gaussian_gp_predict(*args)
                    mean_value = fmean_func.eval(
                        session=sess, feed_dict={free_vars: m.get_free_state()})[0, 0]
                    var_value = fvar_func.eval(
                        session=sess, feed_dict={free_vars: m.get_free_state()})[0, 0]
                    self.assertTrue(np.abs(mean_value - self.posteriorMean) < 1e-4)
                    self.assertTrue(np.abs(var_value - self.posteriorVariance) < 1e-4)


class VariationalMultivariateTest(GPflowTestCase):
    def setUp(self):
        self.nDimensions = 3
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn(self.nDimensions, 1)
        self.X = self.rng.randn(self.nDimensions, 1)
        self.Z = self.X.copy()
        self.noiseVariance = 0.5
        self.signalVariance = 1.5
        self.lengthScale = 1.7
        self.oneLatentFunction = 1
        self.lik = gpflow.likelihoods.Gaussian()
        self.lik.variance = self.noiseVariance
        self.q_mean = self.rng.randn(self.nDimensions, self.oneLatentFunction)
        self.q_sqrt_diag = self.rng.rand(self.nDimensions, self.oneLatentFunction)
        self.q_sqrt_full = np.tril(self.rng.rand(self.nDimensions, self.nDimensions))

    def getModel(self, is_diagonal, is_whitened):
        model = gpflow.svgp.SVGP(
            X=self.X, Y=self.Y,
            kern=kernel(kernelVariance=self.signalVariance, lengthScale=self.lengthScale),
            likelihood=self.lik,
            Z=self.Z,
            q_diag=is_diagonal,
            whiten=is_whitened)
        if is_diagonal:
            model.q_sqrt = self.q_sqrt_diag
        else:
            model.q_sqrt = self.q_sqrt_full[:, :, None]
        model.q_mu = self.q_mean
        return model

    def test_refrence_implementation_consistency(self):
        with self.test_session():
            rng = np.random.RandomState(10)
            qMean = rng.randn()
            qCov = rng.rand()
            pMean = rng.rand()
            pCov = rng.rand()
            univariate_KL = referenceUnivariatePriorKL(qMean, pMean, qCov, pCov)
            multivariate_KL = referenceMultivariatePriorKL(
                np.array([[qMean]]), np.array([[qCov]]),
                np.array([[pMean]]), np.array([[pCov]]))
            self.assertTrue(np.abs(univariate_KL - multivariate_KL) < 1e-4)

    def test_prior_KL_fullQ(self):
        with self.test_session():
            covQ = np.dot(self.q_sqrt_full, self.q_sqrt_full.T)
            mean_prior = np.zeros((self.nDimensions, 1))
            for is_whitened in [True, False]:
                m = self.getModel(False, is_whitened)
                if is_whitened:
                    cov_prior = np.eye(self.nDimensions)
                else:
                    cov_prior = referenceRbfKernel(
                        self.X, self.lengthScale, self.signalVariance)
                referenceKL = referenceMultivariatePriorKL(
                    self.q_mean, covQ, mean_prior, cov_prior)
                # now get test KL.
                test_prior_KL = gpflow.param.AutoFlow()(m.build_prior_KL.__func__)(m)
                self.assertTrue(np.abs(referenceKL - test_prior_KL) < 1e-4)

if __name__ == "__main__":
    unittest.main()
