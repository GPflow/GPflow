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
from numpy.testing import assert_allclose

import gpflow
from gpflow.test_util import GPflowTestCase

from .reference import referenceRbfKernel


def univariate_log_marginal_likelihood(y, K, noiseVariance):
    return (-0.5 * y * y / (K + noiseVariance)
            -0.5 * np.log(K + noiseVariance)
            -0.5 * np.log(np.pi * 2.))


def univariate_posterior(y, K, noiseVariance):
    mean = K * y / (K + noiseVariance)
    variance = K - K / (K + noiseVariance)
    return mean, variance


def univariate_prior_KL(meanA, meanB, varA, varB):
    # KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and qB are univariate normal distributions.
    return (0.5 * (np.log(varB) - np.log(varA) - 1. + varA/varB +
                   (meanB-meanA) * (meanB - meanA) / varB))


def multivariate_prior_KL(meanA, covA, meanB, covB):
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
    return (traceTerm +
            mahalanobisTerm +
            constantTerm +
            priorLogDeterminantTerm +
            variationalLogDeterminantTerm)


def kernel(kernelVariance=1, lengthScale=1.):
    kern = gpflow.kernels.RBF(1)
    kern.variance = kernelVariance
    kern.lengthscales = lengthScale
    return kern


class VariationalUnivariateTest(GPflowTestCase):

    y_real = 2.
    K = 1.
    noiseVariance = 0.5
    univariate = 1
    oneLatentFunction = 1
    meanZero = 0.
    X = np.atleast_2d(np.array([0.]))
    Y = np.atleast_2d(np.array([y_real]))
    Z = X.copy()
    posteriorMean, posteriorVariance = univariate_posterior(
        y=y_real, K=K, noiseVariance=noiseVariance)
    posteriorStd = np.sqrt(posteriorVariance)

    def likelihood(self):
        return gpflow.likelihoods.Gaussian(variance=self.noiseVariance)

    def get_model(self, is_diagonal, is_whitened):
        m = gpflow.models.SVGP(
            X=self.X, Y=self.Y,
            kern=kernel(kernelVariance=self.K),
            likelihood=self.likelihood(),
            Z=self.Z,
            q_diag=is_diagonal,
            whiten=is_whitened,
            autobuild=False)

        if is_diagonal:
            ones = np.ones((self.univariate, self.univariate, self.oneLatentFunction))
            m.q_sqrt = ones * self.posteriorStd
        else:
            ones = np.ones((self.univariate, self.univariate, self.oneLatentFunction))
            m.q_sqrt = ones * self.posteriorStd

        m.q_mu = np.ones((self.univariate, self.oneLatentFunction)) * self.posteriorMean
        m.compile()
        return m

    def test_prior_KL(self):
        with self.test_context():
            meanA = self.posteriorMean
            varA = self.posteriorVariance
            meanB = self.meanZero  # Assumes a zero
            varB = self.K

            referenceKL = univariate_prior_KL(meanA, meanB, varA, varB)

            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    m = self.get_model(is_diagonal, is_whitened)

                    test_prior_KL = gpflow.autoflow()(m.build_prior_KL.__func__)(m)
                    assert_allclose(referenceKL - test_prior_KL, 0, atol=4)

    def test_build_likelihood(self):
        with self.test_context():
            # reference marginal likelihood
            log_marginal_likelihood = univariate_log_marginal_likelihood(
                y=self.y_real, K=self.K, noiseVariance=self.noiseVariance)

            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    model = self.get_model(is_diagonal, is_whitened)
                    model_likelihood = model.compute_log_likelihood()
                    assert_allclose(model_likelihood - log_marginal_likelihood, 0, atol=4)

    def testUnivariateConditionals(self):
        with self.test_context() as sess:
            for is_diagonal in [True, False]:
                for is_whitened in [True, False]:
                    m = self.get_model(is_diagonal, is_whitened)
                    with gpflow.params_as_tensors_for(m):
                        if is_whitened:
                            fmean_func, fvar_func = gpflow.conditionals.conditional(
                                self.X, self.Z, m.kern, m.q_mu, q_sqrt=m.q_sqrt)
                        else:
                            fmean_func, fvar_func = gpflow.conditionals.conditional(
                                self.X, self.Z, m.kern, m.q_mu, q_sqrt=m.q_sqrt, white=True)
                    mean_value = fmean_func.eval(session=sess)[0, 0]
                    var_value = fvar_func.eval(session=sess)[0, 0]
                    assert_allclose(mean_value - self.posteriorMean, 0, atol=4)
                    assert_allclose(var_value - self.posteriorVariance, 0, atol=4)


class VariationalMultivariateTest(GPflowTestCase):

    nDimensions = 3
    rng = np.random.RandomState(1)
    rng = rng
    Y = rng.randn(nDimensions, 1)
    X = rng.randn(nDimensions, 1)
    Z = X.copy()
    noiseVariance = 0.5
    signalVariance = 1.5
    lengthScale = 1.7
    oneLatentFunction = 1
    q_mean = rng.randn(nDimensions, oneLatentFunction)
    q_sqrt_diag = rng.rand(nDimensions, oneLatentFunction)
    q_sqrt_full = np.tril(rng.rand(nDimensions, nDimensions))

    def likelihood(self):
        return gpflow.likelihoods.Gaussian(self.noiseVariance)

    def get_model(self, is_diagonal, is_whitened):
        m = gpflow.models.SVGP(
            X=self.X, Y=self.Y,
            kern=kernel(kernelVariance=self.signalVariance, lengthScale=self.lengthScale),
            likelihood=self.likelihood(),
            Z=self.Z,
            q_diag=is_diagonal,
            whiten=is_whitened)
        if is_diagonal:
            m.q_sqrt = self.q_sqrt_diag
        else:
            m.q_sqrt = self.q_sqrt_full[None, :, :]
        m.q_mu = self.q_mean
        return m

    def test_refrence_implementation_consistency(self):
        with self.test_context():
            rng = np.random.RandomState(10)
            qMean = rng.randn()
            qCov = rng.rand()
            pMean = rng.rand()
            pCov = rng.rand()
            univariate_KL = univariate_prior_KL(qMean, pMean, qCov, pCov)
            multivariate_KL = multivariate_prior_KL(
                np.array([[qMean]]), np.array([[qCov]]),
                np.array([[pMean]]), np.array([[pCov]]))
            assert_allclose(univariate_KL - multivariate_KL, 0, atol=4)

    def test_prior_KL_fullQ(self):
        with self.test_context():
            covQ = np.dot(self.q_sqrt_full, self.q_sqrt_full.T)
            mean_prior = np.zeros((self.nDimensions, 1))
            for is_whitened in [True, False]:
                m = self.get_model(False, is_whitened)
                if is_whitened:
                    cov_prior = np.eye(self.nDimensions)
                else:
                    cov_prior = referenceRbfKernel(
                        self.X, self.lengthScale, self.signalVariance)
                referenceKL = multivariate_prior_KL(
                    self.q_mean, covQ, mean_prior, cov_prior)
                # now get test KL.
                test_prior_KL = gpflow.autoflow()(m.build_prior_KL.__func__)(m)
                assert_allclose(referenceKL - test_prior_KL, 0, atol=4)

if __name__ == "__main__":
    tf.test.main()
