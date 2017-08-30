from __future__ import print_function
import gpflow
import numpy as np
import unittest
import tensorflow as tf

from testing.gpflow_testcase import GPflowTestCase
from nose.plugins.attrib import attr


@attr(speed='slow')
class TestEquivalence(GPflowTestCase):
    """
    Here we make sure the coregionalized model with diagonal coregion kernel and
    with fixed lengthscale is equivalent with normal GP regression.
    """
    def setUp(self):
        with self.test_session():
            rng = np.random.RandomState(0)
            X = [rng.rand(10, 2)*10, rng.rand(20, 2)*10]
            Y = [np.sin(x) + 0.9 * np.cos(x*1.6) + rng.randn(*x.shape) * 0.8 for x in X]
            label = [np.zeros((10, 1)), np.ones((20, 1))]
            perm = list(range(30))
            rng.shuffle(perm)
            self.Xtest = rng.rand(10, 2)*10

            X_augumented = np.hstack([np.concatenate(X), np.concatenate(label)])
            Y_augumented = np.hstack([np.concatenate(Y), np.concatenate(label)])

            # two independent vgps for two sets of data
            k0 = gpflow.kernels.RBF(2)
            k0.lengthscales.fixed = True
            self.vgp0 = gpflow.vgp.VGP(X[0], Y[0], kern=k0,
                                       mean_function=gpflow.mean_functions.Constant(),
                                       likelihood=gpflow.likelihoods.Gaussian())
            k1 = gpflow.kernels.RBF(2)
            k1.lengthscales.fixed = True
            self.vgp1 = gpflow.vgp.VGP(X[1], Y[1], kern=k1,
                                       mean_function=gpflow.mean_functions.Constant(),
                                       likelihood=gpflow.likelihoods.Gaussian())
            # coregionalized gpr
            lik = gpflow.likelihoods.SwitchedLikelihood(
                [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

            kc = gpflow.kernels.RBF(2)
            kc.fixed = True  # lengthscale and variance is fixed.
            coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[2])
            coreg.W.fixed = True

            mean_c = gpflow.mean_functions.SwitchedMeanFunction(
                [gpflow.mean_functions.Constant(), gpflow.mean_functions.Constant()])
            self.cvgp = gpflow.vgp.VGP(X_augumented, Y_augumented,
                                       kern=kc*coreg,
                                       mean_function=mean_c,
                                       likelihood=lik,
                                       num_latent=2)

            self.vgp0.optimize(disp=False, maxiter=300)
            self.vgp1.optimize(disp=False, maxiter=300)
            self.cvgp.optimize(disp=False, maxiter=300)

    def test_all(self):
        with self.test_session():
            # check variance
            self.assertTrue(np.allclose(
                self.vgp0.likelihood.variance.value,
                self.cvgp.likelihood.likelihood_list[0].variance.value,
                atol=1e-2))
            self.assertTrue(np.allclose(
                self.vgp1.likelihood.variance.value,
                self.cvgp.likelihood.likelihood_list[1].variance.value,
                atol=1e-2))

            # check kernel variance
            self.assertTrue(np.allclose(
                self.vgp0.kern.variance.value,
                self.cvgp.kern.coregion.kappa.value[0],
                atol=1.0e-2))
            self.assertTrue(np.allclose(
                self.vgp1.kern.variance.value,
                self.cvgp.kern.coregion.kappa.value[1],
                atol=1.0e-2))

            # check mean values
            self.assertTrue(np.allclose(
                self.vgp0.mean_function.c.value,
                self.cvgp.mean_function.meanfunction_list[0].c.value,
                atol=1.0e-2))
            self.assertTrue(np.allclose(
                self.vgp1.mean_function.c.value,
                self.cvgp.mean_function.meanfunction_list[1].c.value,
                atol=1.0e-2))

            X_augumented0 = np.hstack([self.Xtest, np.zeros((self.Xtest.shape[0], 1))])
            X_augumented1 = np.hstack([self.Xtest, np.ones((self.Xtest.shape[0], 1))])
            Ytest = [np.sin(x) + 0.9 * np.cos(x*1.6) for x in self.Xtest]
            Y_augumented0 = np.hstack([Ytest, np.zeros((self.Xtest.shape[0], 1))])
            Y_augumented1 = np.hstack([Ytest, np.ones((self.Xtest.shape[0], 1))])

            # check predict_f
            pred_f0 = self.vgp0.predict_f(self.Xtest)
            pred_fc0 = self.cvgp.predict_f(X_augumented0)
            self.assertTrue(np.allclose(pred_f0, pred_fc0, atol=1.0e-2))
            pred_f1 = self.vgp1.predict_f(self.Xtest)
            pred_fc1 = self.cvgp.predict_f(X_augumented1)
            self.assertTrue(np.allclose(pred_f1, pred_fc1, atol=1.0e-2))

            # check predict y
            pred_y0 = self.vgp0.predict_y(self.Xtest)
            pred_yc0 = self.cvgp.predict_y(np.hstack([self.Xtest, np.zeros((self.Xtest.shape[0], 1))]))

            # predict_y returns results for all the likelihodds in multi_likelihood
            self.assertTrue(np.allclose(pred_y0[0], pred_yc0[0][:, :np.array(Ytest).shape[1]], atol=1.0e-2))
            self.assertTrue(np.allclose(pred_y0[1], pred_yc0[1][:, :np.array(Ytest).shape[1]], atol=1.0e-2))
            pred_y1 = self.vgp1.predict_y(self.Xtest)
            pred_yc1 = self.cvgp.predict_y(np.hstack([self.Xtest, np.ones((self.Xtest.shape[0], 1))]))
            # predict_y returns results for all the likelihodds in multi_likelihood
            self.assertTrue(np.allclose(pred_y1[0], pred_yc1[0][:, np.array(Ytest).shape[1]:], atol=1.0e-2))
            self.assertTrue(np.allclose(pred_y1[1], pred_yc1[1][:, np.array(Ytest).shape[1]:], atol=1.0e-2))

            # check predict_density
            pred_ydensity0 = self.vgp0.predict_density(self.Xtest, Ytest)
            pred_ydensity_c0 = self.cvgp.predict_density(X_augumented0, Y_augumented0)
            self.assertTrue(np.allclose(pred_ydensity0, pred_ydensity_c0, atol=1e-2))
            pred_ydensity1 = self.vgp1.predict_density(self.Xtest, Ytest)
            pred_ydensity_c1 = self.cvgp.predict_density(X_augumented1, Y_augumented1)
            self.assertTrue(np.allclose(pred_ydensity1, pred_ydensity_c1, atol=1e-2))

            # just check predict_f_samples(self) works
            self.cvgp.predict_f_samples(X_augumented0, 1)
            self.cvgp.predict_f_samples(X_augumented1, 1)

            # check predict_f_full_cov
            self.vgp0.predict_f_full_cov(self.Xtest)
            self.cvgp.predict_f_full_cov(X_augumented0)
            self.vgp1.predict_f_full_cov(self.Xtest)
            self.cvgp.predict_f_full_cov(X_augumented1)


if __name__ == '__main__':
    unittest.main()
