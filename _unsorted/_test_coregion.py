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
import numpy as np
from numpy.testing import assert_allclose

import gpflow
from gpflow.mean_functions import Constant

rng = np.random.RandomState(0)


class Datum:
    X = [rng.rand(10, 2) * 10, rng.rand(20, 2) * 10]
    Y = [np.sin(x) + 0.9 * np.cos(x * 1.6) + rng.randn(*x.shape) * 0.8 for x in X]
    label = [np.zeros((10, 1)), np.ones((20, 1))]
    perm = list(range(30))
    rng.shuffle(perm)
    Xtest = rng.rand(10, 2) * 10
    X_augumented = np.hstack([np.concatenate(X), np.concatenate(label)])
    Y_augumented = np.hstack([np.concatenate(Y), np.concatenate(label)])
    # For predict tests
    X_augumented0 = np.hstack([Xtest, np.zeros((Xtest.shape[0], 1))])
    X_augumented1 = np.hstack([Xtest, np.ones((Xtest.shape[0], 1))])
    Ytest = [np.sin(x) + 0.9 * np.cos(x * 1.6) for x in Xtest]
    Y_augumented0 = np.hstack([Ytest, np.zeros((Xtest.shape[0], 1))])
    Y_augumented1 = np.hstack([Ytest, np.ones((Xtest.shape[0], 1))])


def _prepare_models():
    """
    Prepare models to make sure the coregionalized model with diagonal coregion kernel and
    with fixed lengthscale is equivalent with normal GP regression.
    """
    # 1. Two independent VGPs for two sets of data
    k0 = gpflow.kernels.SquaredExponential()
    k0.lengthscale.trainable = False
    k1 = gpflow.kernels.SquaredExponential()
    k1.lengthscale.trainable = False
    vgp0 = gpflow.models.VGP((Datum.X[0], Datum.Y[0]),
                             kernel=k0,
                             mean_function=Constant(),
                             likelihood=gpflow.likelihoods.Gaussian(), num_latent=2)
    vgp1 = gpflow.models.VGP((Datum.X[1], Datum.Y[1]),
                             kernel=k1,
                             mean_function=Constant(),
                             likelihood=gpflow.likelihoods.Gaussian(), num_latent=2)
    # 2. Coregionalized GPR
    kc = gpflow.kernels.SquaredExponential()
    kc.trainable = False  # lengthscale and variance is fixed.
    coreg = gpflow.kernels.Coregion(output_dim=2, rank=1, active_dims=[2])
    coreg.W.trainable = False
    lik = gpflow.likelihoods.SwitchedLikelihood([gpflow.likelihoods.Gaussian(),
                                                 gpflow.likelihoods.Gaussian()]
                                                )
    mean_c = gpflow.mean_functions.SwitchedMeanFunction(
        [gpflow.mean_functions.Constant(), gpflow.mean_functions.Constant()])
    cvgp = gpflow.models.VGP((Datum.X_augumented, Datum.Y_augumented),
                             kernel=kc * coreg,
                             mean_function=mean_c,
                             likelihood=lik,
                             num_latent=2
                             )

    # Train them for a small number of iterations

    opt = gpflow.optimizers.Scipy()

    def vgp0_closure():
        return vgp0.neg_log_marginal_likelihood()

    def vgp1_closure():
        return vgp1.neg_log_marginal_likelihood()

    def cvgp_closure():
        return cvgp.neg_log_marginal_likelihood()

    opt.minimize(vgp0_closure, variables=vgp0.trainable_variables, options=dict(maxiter=1000))
    opt.minimize(vgp1_closure, variables=vgp1.trainable_variables, options=dict(maxiter=1000))
    opt.minimize(cvgp_closure, variables=cvgp.trainable_variables, options=dict(maxiter=1000))

    return vgp0, vgp1, cvgp


# ------------------------------------------
# Tests
# ------------------------------------------

def test_likelihood_variance():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(vgp0.likelihood.variance.read_value(),
                    cvgp.likelihood.likelihoods[0].variance.read_value(),
                    atol=1e-2)
    assert_allclose(vgp1.likelihood.variance.read_value(),
                    cvgp.likelihood.likelihoods[1].variance.read_value(),
                    atol=1e-2)


def test_kernel_variance():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(vgp0.kernel.variance.read_value(),
                    cvgp.kernel.kernels[1].kappa.read_value()[0],
                    atol=1.0e-2)
    assert_allclose(vgp1.kernel.variance.read_value(),
                    cvgp.kernel.kernels[1].kappa.read_value()[1],
                    atol=1.0e-2)


def test_mean_values():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(vgp0.mean_function.c.read_value(),
                    cvgp.mean_function.meanfunctions[0].c.read_value(),
                    atol=1.0e-2)
    assert_allclose(vgp1.mean_function.c.read_value(),
                    cvgp.mean_function.meanfunctions[1].c.read_value(),
                    atol=1.0e-2)


def test_predict_f():
    vgp0, vgp1, cvgp = _prepare_models()

    pred_f0 = vgp0.predict_f(Datum.Xtest)
    pred_fc0 = cvgp.predict_f(Datum.X_augumented0)
    assert_allclose(pred_f0, pred_fc0, atol=1.0e-2)
    pred_f1 = vgp1.predict_f(Datum.Xtest)
    pred_fc1 = cvgp.predict_f(Datum.X_augumented1)
    assert_allclose(pred_f1, pred_fc1, atol=1.0e-2)

    # check predict_f_full_cov
    vgp0.predict_f_full_cov(Datum.Xtest)
    cvgp.predict_f_full_cov(Datum.X_augumented0)
    vgp1.predict_f_full_cov(Datum.Xtest)
    cvgp.predict_f_full_cov(Datum.X_augumented1)


def test_predict_y():
    vgp0, vgp1, cvgp = _prepare_models()
    pred_y0 = vgp0.predict_y(Datum.Xtest)
    pred_yc0 = cvgp.predict_y(
        np.hstack([Datum.Xtest, np.zeros((Datum.Xtest.shape[0], 1))]))

    # predict_y returns results for all the likelihodds in multi_likelihood
    assert_allclose(pred_y0[0], pred_yc0[0][:, :np.array(Datum.Ytest).shape[1]], atol=1.0e-2)
    assert_allclose(pred_y0[1], pred_yc0[1][:, :np.array(Datum.Ytest).shape[1]], atol=1.0e-2)
    pred_y1 = vgp1.predict_y(Datum.Xtest)
    pred_yc1 = cvgp.predict_y(
        np.hstack([Datum.Xtest, np.ones((Datum.Xtest.shape[0], 1))]))

    # predict_y returns results for all the likelihodds in multi_likelihood
    assert_allclose(pred_y1[0], pred_yc1[0][:, np.array(Datum.Ytest).shape[1]:], atol=1.0e-2)
    assert_allclose(pred_y1[1], pred_yc1[1][:, np.array(Datum.Ytest).shape[1]:], atol=1.0e-2)


def test_predict_log_density():
    vgp0, vgp1, cvgp = _prepare_models()

    pred_ydensity0 = vgp0.predict_log_density((Datum.Xtest, Datum.Ytest))
    pred_ydensity_c0 = cvgp.predict_log_density((Datum.X_augumented0, Datum.Y_augumented0))
    assert_allclose(pred_ydensity0, pred_ydensity_c0, atol=1e-2)
    pred_ydensity1 = vgp1.predict_log_density((Datum.Xtest, Datum.Ytest))
    pred_ydensity_c1 = cvgp.predict_log_density((Datum.X_augumented1, Datum.Y_augumented1))
    assert_allclose(pred_ydensity1, pred_ydensity_c1, atol=1e-2)


def test_predict_f_samples():
    vgp0, vgp1, cvgp = _prepare_models()
    # just check predict_f_samples(self) works
    cvgp.predict_f_samples(Datum.X_augumented0, 1)
    cvgp.predict_f_samples(Datum.X_augumented1, 1)
