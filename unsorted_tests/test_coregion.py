# Copyright 2017 the GPflow authors.
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


class TestEquivalence(GPflowTestCase):
    """
    Here we make sure the coregionalized model with diagonal coregion kernel and
    with fixed lengthscale is equivalent with normal GP regression.
    """
    def prepare(self):
        rng = np.random.RandomState(0)
        X = [rng.rand(10, 2) * 10, rng.rand(20, 2) * 10]
        Y = [np.sin(x) + 0.9 * np.cos(x * 1.6) + rng.randn(*x.shape) * 0.8 for x in X]
        label = [np.zeros((10, 1)), np.ones((20, 1))]
        perm = list(range(30))
        rng.shuffle(perm)
        Xtest = rng.rand(10, 2) * 10

        X_augumented = np.hstack([np.concatenate(X), np.concatenate(label)])
        Y_augumented = np.hstack([np.concatenate(Y), np.concatenate(label)])

        # 1. Two independent VGPs for two sets of data

        k0 = gpflow.kernels.RBF(2)
        k0.lengthscales.trainable = False
        vgp0 = gpflow.models.VGP(
            X[0], Y[0], kern=k0,
            mean_function=gpflow.mean_functions.Constant(),
            likelihood=gpflow.likelihoods.Gaussian())
        k1 = gpflow.kernels.RBF(2)
        k1.lengthscales.trainable = False
        vgp1 = gpflow.models.VGP(
            X[1], Y[1], kern=k1,
            mean_function=gpflow.mean_functions.Constant(),
            likelihood=gpflow.likelihoods.Gaussian())

        # 2. Coregionalized GPR

        lik = gpflow.likelihoods.SwitchedLikelihood(
            [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()])

        kc = gpflow.kernels.RBF(2)
        kc.trainable = False  # lengthscale and variance is fixed.
        coreg = gpflow.kernels.Coregion(1, output_dim=2, rank=1, active_dims=[2])
        coreg.W.trainable = False

        mean_c = gpflow.mean_functions.SwitchedMeanFunction(
            [gpflow.mean_functions.Constant(), gpflow.mean_functions.Constant()])
        cvgp = gpflow.models.VGP(
            X_augumented, Y_augumented,
            kern=kc * coreg,
            mean_function=mean_c,
            likelihood=lik,
            num_latent=2)
        return vgp0, vgp1, cvgp, Xtest

    def setup(self):
        vgp0, vgp1, cvgp, Xtest = self.prepare()
        opt1 = gpflow.train.ScipyOptimizer()
        opt2 = gpflow.train.ScipyOptimizer()
        opt3 = gpflow.train.ScipyOptimizer()
        opt1.minimize(vgp0, maxiter=50)
        opt2.minimize(vgp1, maxiter=50)
        opt3.minimize(cvgp, maxiter=50)
        self.Xtest = Xtest
        self.vgp0 = vgp0
        self.vgp1 = vgp1
        self.cvgp = cvgp

    def test_likelihood_variance(self):
        with self.test_context():
            self.setup()
            assert_allclose(self.vgp0.likelihood.variance.read_value(),
                            self.cvgp.likelihood.likelihood_list[0].variance.read_value(),
                            atol=1e-2)
            assert_allclose(self.vgp1.likelihood.variance.read_value(),
                            self.cvgp.likelihood.likelihood_list[1].variance.read_value(),
                            atol=1e-2)

    def test_kernel_variance(self):
        with self.test_context():
            self.setup()
            assert_allclose(self.vgp0.kern.variance.read_value(),
                            self.cvgp.kern.kernels[1].kappa.read_value()[0],
                            atol=1.0e-2)
            assert_allclose(self.vgp1.kern.variance.read_value(),
                            self.cvgp.kern.kernels[1].kappa.read_value()[1],
                            atol=1.0e-2)

    def test_mean_values(self):
        with self.test_context():
            self.setup()
            assert_allclose(self.vgp0.mean_function.c.read_value(),
                            self.cvgp.mean_function.meanfunction_list[0].c.read_value(),
                            atol=1.0e-2)
            assert_allclose(self.vgp1.mean_function.c.read_value(),
                            self.cvgp.mean_function.meanfunction_list[1].c.read_value(),
                            atol=1.0e-2)

    def test_predicts(self):
        with self.test_context():
            self.setup()
            X_augumented0 = np.hstack([self.Xtest, np.zeros((self.Xtest.shape[0], 1))])
            X_augumented1 = np.hstack([self.Xtest, np.ones((self.Xtest.shape[0], 1))])
            Ytest = [np.sin(x) + 0.9 * np.cos(x*1.6) for x in self.Xtest]
            Y_augumented0 = np.hstack([Ytest, np.zeros((self.Xtest.shape[0], 1))])
            Y_augumented1 = np.hstack([Ytest, np.ones((self.Xtest.shape[0], 1))])

            # check predict_f
            pred_f0 = self.vgp0.predict_f(self.Xtest)
            pred_fc0 = self.cvgp.predict_f(X_augumented0)
            assert_allclose(pred_f0, pred_fc0, atol=1.0e-2)
            pred_f1 = self.vgp1.predict_f(self.Xtest)
            pred_fc1 = self.cvgp.predict_f(X_augumented1)
            assert_allclose(pred_f1, pred_fc1, atol=1.0e-2)

            # check predict y
            pred_y0 = self.vgp0.predict_y(self.Xtest)
            pred_yc0 = self.cvgp.predict_y(
                np.hstack([self.Xtest, np.zeros((self.Xtest.shape[0], 1))]))

            # predict_y returns results for all the likelihodds in multi_likelihood
            assert_allclose(pred_y0[0], pred_yc0[0][:, :np.array(Ytest).shape[1]], atol=1.0e-2)
            assert_allclose(pred_y0[1], pred_yc0[1][:, :np.array(Ytest).shape[1]], atol=1.0e-2)
            pred_y1 = self.vgp1.predict_y(self.Xtest)
            pred_yc1 = self.cvgp.predict_y(
                np.hstack([self.Xtest, np.ones((self.Xtest.shape[0], 1))]))

            # predict_y returns results for all the likelihodds in multi_likelihood
            assert_allclose(pred_y1[0], pred_yc1[0][:, np.array(Ytest).shape[1]:], atol=1.0e-2)
            assert_allclose(pred_y1[1], pred_yc1[1][:, np.array(Ytest).shape[1]:], atol=1.0e-2)

            # check predict_density
            pred_ydensity0 = self.vgp0.predict_density(self.Xtest, Ytest)
            pred_ydensity_c0 = self.cvgp.predict_density(X_augumented0, Y_augumented0)
            self.assertTrue(np.allclose(pred_ydensity0, pred_ydensity_c0, atol=1e-2))
            pred_ydensity1 = self.vgp1.predict_density(self.Xtest, Ytest)
            pred_ydensity_c1 = self.cvgp.predict_density(X_augumented1, Y_augumented1)
            np.testing.assert_allclose(pred_ydensity1, pred_ydensity_c1, atol=1e-2)

            # just check predict_f_samples(self) works
            self.cvgp.predict_f_samples(X_augumented0, 1)
            self.cvgp.predict_f_samples(X_augumented1, 1)

            # check predict_f_full_cov
            self.vgp0.predict_f_full_cov(self.Xtest)
            self.cvgp.predict_f_full_cov(X_augumented0)
            self.vgp1.predict_f_full_cov(self.Xtest)
            self.cvgp.predict_f_full_cov(X_augumented1)


if __name__ == '__main__':
    tf.test.main()
