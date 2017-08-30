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

import numpy as np
import tensorflow as tf
from nose.plugins.attrib import attr
import unittest
import gpflow

from testing.gpflow_testcase import GPflowTestCase


@attr(speed='slow')
class TestEquivalence(GPflowTestCase):
    """
    With a Gaussian likelihood, and inducing points (where appropriate)
    positioned at the data, many of the gpflow methods are equivalent (perhaps
    subject to some optimization).

    Here, we make 5 models that should be the same, and make sure some
    similarites hold. The models are:

    1) GP Regression
    2) Variational GP (with the likelihood set to Gaussian)
    3) Sparse variational GP (likelihood is Gaussian, inducing points
       at the data)
    4) Sparse variational GP (as above, but with the whitening rotation
       of the inducing variables)
    5) Sparse variational GP Regression (as above, but there the inducing
       variables are 'collapsed' out, as in Titsias 2009)
    """

    def setUp(self):
        with self.test_session():
            rng = np.random.RandomState(0)
            X = rng.rand(20, 1) * 10
            Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
            Y = np.tile(Y, 2)  # two identical columns
            self.Xtest = rng.rand(10, 1) * 10

            m1 = gpflow.gpr.GPR(
                X, Y, kern=gpflow.kernels.RBF(1),
                mean_function=gpflow.mean_functions.Constant())
            m2 = gpflow.vgp.VGP(
                X, Y, gpflow.kernels.RBF(1), likelihood=gpflow.likelihoods.Gaussian(),
                mean_function=gpflow.mean_functions.Constant())
            m3 = gpflow.svgp.SVGP(
                X, Y, gpflow.kernels.RBF(1),
                likelihood=gpflow.likelihoods.Gaussian(),
                Z=X.copy(),
                q_diag=False,
                mean_function=gpflow.mean_functions.Constant())
            m3.Z.fixed = True
            m4 = gpflow.svgp.SVGP(
                X, Y, gpflow.kernels.RBF(1),
                likelihood=gpflow.likelihoods.Gaussian(),
                Z=X.copy(), q_diag=False, whiten=True,
                mean_function=gpflow.mean_functions.Constant())
            m4.Z.fixed = True
            m5 = gpflow.sgpr.SGPR(
                X, Y, gpflow.kernels.RBF(1),
                Z=X.copy(),
                mean_function=gpflow.mean_functions.Constant())

            m5.Z.fixed = True
            m6 = gpflow.sgpr.GPRFITC(
                X, Y, gpflow.kernels.RBF(1), Z=X.copy(),
                mean_function=gpflow.mean_functions.Constant())
            m6.Z.fixed = True
            self.models = [m1, m2, m3, m4, m5, m6]
            for m in self.models:
                m.optimize(disp=False, maxiter=300)
                # TODO(awav): Instead of this hack the test must be splitted up
                print('.')  # stop travis timing out

    def test_all(self):
        with self.test_session():
            likelihoods = np.array([
                -m._objective(m.get_free_state())[0].squeeze()
                for m in self.models])
            self.assertTrue(np.allclose(likelihoods, likelihoods[0], 1e-6))
            variances, lengthscales = [], []
            for m in self.models:
                if hasattr(m.kern, 'rbf'):
                    variances.append(m.kern.rbf.variance.value)
                    lengthscales.append(m.kern.rbf.lengthscales.value)
                else:
                    variances.append(m.kern.variance.value)
                    lengthscales.append(m.kern.lengthscales.value)
            variances, lengthscales = np.array(variances), np.array(lengthscales)
            self.assertTrue(np.allclose(variances, variances[0], 1e-5))
            self.assertTrue(np.allclose(lengthscales, lengthscales.mean(), 1e-4))
            mu0, var0 = self.models[0].predict_y(self.Xtest)
            for m in self.models[1:]:
                mu, var = m.predict_y(self.Xtest)
                self.assertTrue(np.allclose(mu, mu0, 1e-3))
                self.assertTrue(np.allclose(var, var0, 1e-4))


class VGPTest(GPflowTestCase):
    def test_vgp_vs_svgp(self):
        with self.test_session():
            N, Ns, DX, DY = 100, 10, 2, 2

            np.random.seed(1)
            X = np.random.randn(N, DX)
            Xs = np.random.randn(Ns, DX)
            Y = np.random.randn(N, DY)

            kern = gpflow.kernels.Matern52(DX)
            likelihood = gpflow.likelihoods.StudentT()

            m_svgp = gpflow.svgp.SVGP(X, Y, kern, likelihood, X.copy(),
                                      whiten=True, q_diag=False)
            m_vgp = gpflow.vgp.VGP(X, Y, kern, likelihood)

            q_mu = np.random.randn(N, DY)
            q_sqrt = np.random.randn(N, N, DY)

            m_svgp.q_mu = q_mu
            m_svgp.q_sqrt = q_sqrt

            m_vgp.q_mu = q_mu
            m_vgp.q_sqrt = q_sqrt

            L_svgp = m_svgp.compute_log_likelihood()
            L_vgp = m_vgp.compute_log_likelihood()
            assert np.allclose(L_svgp, L_vgp)

            pred_svgp = m_svgp.predict_f(Xs)
            pred_vgp = m_vgp.predict_f(Xs)
            assert np.allclose(pred_svgp[0], pred_vgp[0])
            assert np.allclose(pred_svgp[1], pred_vgp[1])

    def test_vgp_vs_opper_archambeau(self):
        with self.test_session():
            N, Ns, DX, DY = 100, 10, 2, 2

            np.random.seed(1)
            X = np.random.randn(N, DX)
            Xs = np.random.randn(Ns, DX)
            Y = np.random.randn(N, DY)

            kern = gpflow.kernels.Matern52(DX)
            likelihood = gpflow.likelihoods.StudentT()

            m_vgp = gpflow.vgp.VGP(X, Y, kern, likelihood)

            m_vgp_oa = gpflow.vgp.VGP_opper_archambeau(X, Y, kern, likelihood)

            q_alpha = np.random.randn(N, DX)
            q_lambda = np.random.randn(N, DX) ** 2

            m_vgp_oa.q_alpha = q_alpha
            m_vgp_oa.q_lambda = q_lambda

            K = kern.compute_K_symm(X) + np.eye(N) * gpflow._settings.settings.numerics.jitter_level
            L = np.linalg.cholesky(K)
            L_inv = np.linalg.inv(L)
            K_inv = np.linalg.inv(K)

            mean = K.dot(q_alpha)
            prec_dnn = K_inv[None, :, :] + np.array([np.diag(l ** 2) for l in q_lambda.T])
            var_dnn = np.linalg.inv(prec_dnn)

            m_svgp_unwhitened = gpflow.svgp.SVGP(
                X, Y, kern, likelihood, X.copy(),
                whiten=False, q_diag=False)

            m_svgp_unwhitened.q_mu = mean
            m_svgp_unwhitened.q_sqrt = np.transpose(np.linalg.cholesky(var_dnn), [1, 2, 0])

            mean_white_nd = L_inv.dot(mean)
            var_white_dnn = np.einsum('nN,dNM,mM->dnm', L_inv, var_dnn, L_inv)

            q_sqrt_nnd = np.transpose(np.linalg.cholesky(var_white_dnn), [1, 2, 0])

            m_vgp.q_mu = mean_white_nd
            m_vgp.q_sqrt = q_sqrt_nnd

            L_vgp = m_vgp.compute_log_likelihood()
            L_svgp_unwhitened = m_svgp_unwhitened.compute_log_likelihood()
            L_vgp_oa = m_vgp_oa.compute_log_likelihood()
            assert np.allclose(L_vgp, L_vgp_oa)
            assert np.allclose(L_vgp, L_svgp_unwhitened)

            pred_vgp = m_vgp.predict_f(Xs)
            pred_svgp_unwhitened = m_svgp_unwhitened.predict_f(Xs)
            pred_vgp_oa = m_vgp_oa.predict_f(Xs)

            assert np.allclose(pred_vgp[0], pred_vgp_oa[0])
            assert np.allclose(pred_vgp[0], pred_svgp_unwhitened[0])
            assert np.allclose(pred_vgp[1], pred_vgp_oa[1], rtol=1e-4)  # jitter?
            assert np.allclose(pred_vgp[1], pred_svgp_unwhitened[1], rtol=1e-4)

    def test_recompile(self):
        with self.test_session():
            N, DX, DY = 100, 2, 2

            np.random.seed(1)
            X = np.random.randn(N, DX)
            Y = np.random.randn(N, DY)

            kern = gpflow.kernels.Matern52(DX)
            likelihood = gpflow.likelihoods.StudentT()

            m_vgp = gpflow.vgp.VGP(X, Y, kern, likelihood)
            m_vgp_oa = gpflow.vgp.VGP_opper_archambeau(X, Y, kern, likelihood)

            try:
                for m in [m_vgp, m_vgp_oa]:
                    m.optimize(maxiter=1)
                    m.X = X[:-1, :]
                    m.Y = Y[:-1, :]
                    m.optimize(maxiter=1)
            except:
                assert False, 'array mismatch'


class TestUpperBound(GPflowTestCase):
    """
    Test for upper bound for regression marginal likelihood
    """

    def setUp(self):
        self.X = np.random.rand(100, 1)
        self.Y = np.sin(1.5 * 2 * np.pi * self.X) + np.random.randn(*self.X.shape) * 0.1

    def test_few_inducing_points(self):
        with self.test_session():
            vfe = gpflow.sgpr.SGPR(self.X, self.Y, gpflow.kernels.RBF(1), self.X[:10, :].copy())
            vfe.optimize()

            full = gpflow.gpr.GPR(self.X, self.Y, gpflow.kernels.RBF(1))
            full.kern.lengthscales = vfe.kern.lengthscales.value
            full.kern.variance = vfe.kern.variance.value
            full.likelihood.variance = vfe.likelihood.variance.value
            full.compile()

            lml_upper = vfe.compute_upper_bound()
            lml_vfe = -vfe._objective(vfe.get_free_state())[0]
            lml_full = -full._objective(full.get_free_state())[0]

            self.assertTrue(lml_upper > lml_full > lml_vfe)


if __name__ == '__main__':
    unittest.main()
