import gpflow
import numpy as np
import unittest

from gpflow.test_util import GPflowTestCase
from gpflow import settings
from gpflow.conditionals import conditional, uncertain_conditional

float_type = settings.dtypes.float_type

np.random.seed(0)

class MomentMatchingSVGP(gpflow.models.SVGP):

    @gpflow.decors.params_as_tensors
    @gpflow.decors.autoflow((float_type, [None, None]), (float_type, [None, None, None]))
    def uncertain_predict_f_moment_matching(self, Xmu, Xcov):
        return uncertain_conditional(Xmu, Xcov, self.feat, self.kern, self.q_mu, self.q_sqrt, whiten=self.whiten)

    def uncertain_predict_f_monte_carlo(self, Xmu, Xstd, mc_iter=1000000):
        X_samples = np.random.randn(mc_iter, 1) * (Xstd ** 0.5) + Xmu
        F_mu, F_var = self.predict_f(X_samples)
        F_samples = F_mu + np.random.randn(*F_var.shape) * (F_var ** 0.5)
        mean = np.mean(F_samples, axis=0)
        covar = np.cov(F_samples.T)
        return mean, covar


class NoUncertaintyTest(GPflowTestCase):

    def setUp(self):
        self.N, self.Nnew, self.D_out = 7, 2, 3
        self.X = np.linspace(-5, 5, self.N)[:, None] + np.random.randn(self.N, 1)
        self.Y = np.hstack([np.sin(self.X), np.cos(self.X), self.X**2])
        self.Xnew_mu = np.random.randn(self.Nnew, 1)
        self.Xnew_covar = np.zeros((self.Nnew, 1, 1))

    def test_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=True)
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.predict_f(self.Xnew_mu)
        mean2, var2 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        np.testing.assert_allclose(mean1, mean2)
        for n in range(self.Nnew):
            np.testing.assert_allclose(var1[n,:], np.diag(var2[n,...]))

    def test_non_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=False)
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.predict_f(self.Xnew_mu)
        mean2, var2 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        np.testing.assert_allclose(mean1, mean2)
        for n in range(self.Nnew):
            np.testing.assert_allclose(var1[n,:], np.diag(var2[n,...]))



class MonteCarloTest(GPflowTestCase):

    def setUp(self):
        self.N, self.Nnew, self.D_out = 7, 2, 3
        self.X = np.linspace(-5, 5, self.N)[:, None] + np.random.randn(self.N, 1)
        self.Y = np.hstack([np.sin(self.X), np.cos(self.X) * 2, self.X**2])
        self.Xnew_mu = np.random.randn(self.Nnew, 1)
        self.Xnew_covar = np.random.randn(self.Nnew, 1, 1) ** 2

    def test_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=True)
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        for n in range(self.Nnew):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.Xnew_covar[n,...])
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)

    def test_non_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=False)
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        for n in range(self.Nnew):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.Xnew_covar[n,...])
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)


if __name__ == "__main__":
    unittest.main()
