import gpflow
import tensorflow as tf
import numpy as np
import unittest

from gpflow.test_util import GPflowTestCase
from gpflow import settings
from gpflow.conditionals import conditional, uncertain_conditional, feature_conditional
from gpflow.quadrature import mvnquad

float_type = settings.dtypes.float_type

np.random.seed(0)

class MomentMatchingSVGP(gpflow.models.SVGP):

    @gpflow.decors.params_as_tensors
    @gpflow.decors.autoflow((float_type, [None, None]), (float_type, [None, None, None]))
    def uncertain_predict_f_moment_matching(self, Xmu, Xcov):
        return uncertain_conditional(Xmu, Xcov, self.feat, self.kern, self.q_mu, self.q_sqrt, whiten=self.whiten, full_cov_output=self.full_cov_output)

    def uncertain_predict_f_monte_carlo(self, Xmu, Xchol, mc_iter=1000000):
        D_in = Xchol.shape[0]
        X_samples = Xmu + np.reshape(Xchol[None, :, :] @ np.random.randn(mc_iter, D_in)[:, :, None], [mc_iter, D_in])
        F_mu, F_var = self.predict_f(X_samples)
        F_samples = F_mu + np.random.randn(*F_var.shape) * (F_var ** 0.5)
        mean = np.mean(F_samples, axis=0)
        covar = np.cov(F_samples.T)
        return mean, covar

class NoUncertaintyTest(GPflowTestCase):
    N, N_new, D_out = 7, 2, 3

    def setUp(self):
        self.X = np.linspace(-5, 5, self.N)[:, None] + np.random.randn(self.N, 1)
        self.Y = np.hstack([np.sin(self.X), np.cos(self.X), self.X**2])
        self.Xnew_mu = np.random.randn(self.N_new, 1)
        self.Xnew_covar = np.zeros((self.N_new, 1, 1))

    def test_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=True)
        model.full_cov_output = False
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.predict_f(self.Xnew_mu)
        mean2, var2 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        np.testing.assert_allclose(mean1, mean2)
        for n in range(self.N_new):
            np.testing.assert_allclose(var1[n,:], var2[n,...])

    def test_non_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.randn() ** 2)
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=False)
        model.full_cov_output = False
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.predict_f(self.Xnew_mu)
        mean2, var2 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        np.testing.assert_allclose(mean1, mean2)
        for n in range(self.N_new):
            np.testing.assert_allclose(var1[n,:], var2[n,...])



class MonteCarloTest_1_Din(GPflowTestCase):
    N, N_new, D_out = 7, 2, 3

    def setUp(self):
        self.X = np.linspace(-5, 5, self.N)[:, None] + np.random.randn(self.N, 1)
        self.Y = np.hstack([np.sin(self.X), np.cos(self.X) * 2, self.X**2])
        self.Xnew_mu = np.random.randn(self.N_new, 1)
        self.Xnew_covar = np.random.rand(self.N_new, 1, 1)

    def test_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.rand())
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=True)
        model.full_cov_output = True
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)
        for n in range(self.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.Xnew_covar[n,...] ** 0.5)
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)

    def test_non_whiten(self):
        k = gpflow.ekernels.RBF(1, variance=np.random.rand())
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=False)
        model.full_cov_output = True
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        for n in range(self.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.Xnew_covar[n,...] ** 0.5)
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)


class MonteCarloTest_2_Din(GPflowTestCase):
    N, N_new, D_in, D_out = 7, 5, 3, 4

    def setUp(self):
        self.X = np.random.randn(self.N, self.D_in)
        self.Y = np.random.randn(self.N, self.D_out)
        self.Xnew_mu = np.random.randn(self.N_new, self.D_in)
        self.Xnew_covar, self.L = [], []
        for _ in range(self.N_new):
            L = np.tril(np.random.randn(self.D_in, self.D_in))
            self.L.append(L)
            self.Xnew_covar.append(L @ L.T)
        self.L = np.array(self.L)
        self.Xnew_covar = np.array(self.Xnew_covar)

    def test_whiten(self):
        k = gpflow.ekernels.RBF(self.D_in, variance=np.random.rand())
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=True)
        model.full_cov_output = True
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        for n in range(self.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.L[n,...])
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)

    def test_non_whiten(self):
        k = gpflow.ekernels.RBF(self.D_in, variance=np.random.rand())
        model = MomentMatchingSVGP(self.X, self.Y, k, gpflow.likelihoods.Gaussian(), Z=self.X.copy(), whiten=False)
        model.full_cov_output = True
        model.compile()
        gpflow.train.AdamOptimizer().minimize(model)

        mean1, var1 = model.uncertain_predict_f_moment_matching(self.Xnew_mu, self.Xnew_covar)

        for n in range(self.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(self.Xnew_mu[n,...], self.L[n,...])
            np.testing.assert_almost_equal(mean1[n,...], mean2, decimal=3)
            np.testing.assert_almost_equal(var1[n,...], var2, decimal=2)


class QuadratureTest(GPflowTestCase):
    num_data = 10
    num_ind = 10
    D_in, D_out, H = 2, 3, 150

    def setup(self):

        # Numpy objects with random data
        self.Xmu = np.random.randn(self.num_data, self.D_in)
        self.Xvar = []
        for _ in range(self.num_data):
            L = np.tril(np.random.randn(self.D_in, self.D_in))
            self.Xvar.append(L @ L.T)
        self.Xvar = np.array(self.Xvar)
        self.Z = np.random.randn(self.num_ind, self.D_in)
        self.q_mu = np.random.randn(self.num_ind, self.D_out)
        self.q_sqrt = np.array([np.tril(np.random.randn(self.num_ind, self.num_ind)) for _ in range(self.D_out)])
        self.q_sqrt = np.transpose(self.q_sqrt, [1, 2, 0])

        # Tensorflow placeholders
        self.Xmu_tf = tf.placeholder(float_type, [self.num_data, self.D_in], name="Xmu")
        self.Xvar_tf = tf.placeholder(float_type, [self.num_data, self.D_in, self.D_in], name="Xvar")
        self.q_mu_tf = tf.placeholder(float_type, [self.num_ind, self.D_out], name="q_mu")
        self.q_sqrt_tf = tf.placeholder(float_type, [self.num_ind, self.num_ind, self.D_out])

        self.kern = gpflow.ekernels.RBF(self.D_in)
        self.feat = gpflow.features.InducingPoints(self.Z)
        self.kern.compile()
        self.feat.compile()

        self.feed_dict = \
            {
                self.Xmu_tf:self.Xmu,
                self.Xvar_tf:self.Xvar,
                self.q_mu_tf:self.q_mu,
                self.q_sqrt_tf:self.q_sqrt
            }


    def test_whiten(self):
        with self.test_context() as sess:
            self.setup()
            self.whiten = True

            mean_quad = mvnquad(self.mean_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            var_quad = mvnquad(self.var_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            mean_2_quad = mvnquad(self.mean_2_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            mean_analytic, var_analytic = uncertain_conditional(self.Xmu_tf, self.Xvar_tf, self.feat, self.kern,
                                                self.q_mu_tf, self.q_sqrt_tf, full_cov_output=False, whiten=self.whiten)

            mean_quad, var_quad, mean_2_quad = sess.run([mean_quad, var_quad, mean_2_quad], feed_dict=self.feed_dict)
            var_quad = var_quad + (mean_2_quad - mean_quad**2)
            mean_analytic, var_analytic = sess.run([mean_analytic, var_analytic], feed_dict=self.feed_dict)

            np.testing.assert_almost_equal(mean_quad, mean_analytic, decimal=6)
            np.testing.assert_almost_equal(var_quad, var_analytic, decimal=6)


    def test_non_whiten(self):
        with self.test_context() as sess:
            self.setup()
            self.whiten = False

            mean_quad = mvnquad(self.mean_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            var_quad = mvnquad(self.var_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            mean_2_quad = mvnquad(self.mean_2_func, self.Xmu_tf, self.Xvar_tf, self.H, self.D_in, (self.D_out,))
            mean_analytic, var_analytic = uncertain_conditional(self.Xmu_tf, self.Xvar_tf, self.feat, self.kern,
                                                self.q_mu_tf, self.q_sqrt_tf, full_cov_output=False, whiten=self.whiten)

            mean_quad, var_quad, mean_2_quad = sess.run([mean_quad, var_quad, mean_2_quad], feed_dict=self.feed_dict)
            var_quad = var_quad + (mean_2_quad - mean_quad**2)
            mean_analytic, var_analytic = sess.run([mean_analytic, var_analytic], feed_dict=self.feed_dict)

            np.testing.assert_almost_equal(mean_quad, mean_analytic, decimal=6)
            np.testing.assert_almost_equal(var_quad, var_analytic, decimal=6)


    def mean_func(self, X):
        mean, _  = feature_conditional(X, self.feat, self.kern, self.q_mu_tf, q_sqrt=self.q_sqrt_tf, whiten=self.whiten)
        return mean

    def var_func(self, X):
        _, var = feature_conditional(X, self.feat, self.kern, self.q_mu_tf, q_sqrt=self.q_sqrt_tf, whiten=self.whiten)
        return var

    def mean_2_func(self, X):
        mean, _  = feature_conditional(X, self.feat, self.kern, self.q_mu_tf, q_sqrt=self.q_sqrt_tf, whiten=self.whiten)
        return mean**2

if __name__ == "__main__":
    unittest.main()
