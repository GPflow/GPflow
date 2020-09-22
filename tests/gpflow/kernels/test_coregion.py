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
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow import set_trainable
from gpflow.mean_functions import Constant

rng = np.random.RandomState(0)


class Datum:
    N1, N2 = 6, 16
    X = [rng.rand(N1, 2) * 1, rng.rand(N2, 2) * 1]
    Y = [
        np.sin(x[:, :1]) + 0.9 * np.cos(x[:, 1:2] * 1.6) + rng.randn(x.shape[0], 1) * 0.8 for x in X
    ]
    label = [np.zeros((N1, 1)), np.ones((N2, 1))]
    X_augmented0 = np.hstack([X[0], label[0]])
    X_augmented1 = np.hstack([X[1], label[1]])
    X_augmented = np.vstack([X_augmented0, X_augmented1])

    Y_augmented0 = np.hstack([Y[0], label[0]])
    Y_augmented1 = np.hstack([Y[1], label[1]])
    Y_augmented = np.vstack([Y_augmented0, Y_augmented1])

    # For predict tests
    N = 10
    Xtest = rng.rand(N, 2) * N
    Xtest_augmented0 = np.hstack([Xtest, np.zeros((N, 1))])
    Xtest_augmented1 = np.hstack([Xtest, np.ones((N, 1))])
    Ytest = np.sin(Xtest[:, :1]) + 0.9 * np.cos(Xtest[:, 1:2] * 1.6)
    Ytest_augmented0 = np.hstack([Ytest, np.zeros((N, 1))])
    Ytest_augmented1 = np.hstack([Ytest, np.ones((N, 1))])


def _prepare_models():
    """
    Prepare models to make sure the coregionalized model with diagonal coregion kernel and
    with fixed lengthscales is equivalent with normal GP regression.
    """
    # 1. Two independent VGPs for two sets of data
    k0 = gpflow.kernels.SquaredExponential()
    set_trainable(k0.lengthscales, False)
    k1 = gpflow.kernels.SquaredExponential()
    set_trainable(k1.lengthscales, False)
    vgp0 = gpflow.models.VGP(
        (Datum.X[0], Datum.Y[0]),
        kernel=k0,
        mean_function=Constant(),
        likelihood=gpflow.likelihoods.Gaussian(),
        num_latent_gps=1,
    )
    vgp1 = gpflow.models.VGP(
        (Datum.X[1], Datum.Y[1]),
        kernel=k1,
        mean_function=Constant(),
        likelihood=gpflow.likelihoods.Gaussian(),
        num_latent_gps=1,
    )
    # 2. Coregionalized VGP
    kc = gpflow.kernels.SquaredExponential(active_dims=[0, 1])
    set_trainable(kc.lengthscales, False)
    set_trainable(kc.variance, False)  # variance is handled by the Coregion kernel
    coreg = gpflow.kernels.Coregion(output_dim=2, rank=1, active_dims=[2])
    coreg.W.assign(np.zeros((2, 1)))  # zero correlation between outputs
    set_trainable(coreg.W, False)
    lik = gpflow.likelihoods.SwitchedLikelihood(
        [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Gaussian()]
    )
    mean_c = gpflow.mean_functions.SwitchedMeanFunction(
        [gpflow.mean_functions.Constant(), gpflow.mean_functions.Constant()]
    )
    cvgp = gpflow.models.VGP(
        (Datum.X_augmented, Datum.Y_augmented),
        kernel=kc * coreg,
        mean_function=mean_c,
        likelihood=lik,
        num_latent_gps=1,
    )

    # Train them for a small number of iterations

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        vgp0.training_loss,
        variables=vgp0.trainable_variables,
        options=dict(maxiter=1000),
        method="BFGS",
    )
    opt.minimize(
        vgp1.training_loss,
        variables=vgp1.trainable_variables,
        options=dict(maxiter=1000),
        method="BFGS",
    )
    opt.minimize(
        cvgp.training_loss,
        variables=cvgp.trainable_variables,
        options=dict(maxiter=1000),
        method="BFGS",
    )

    return vgp0, vgp1, cvgp


# ------------------------------------------
# Tests
# ------------------------------------------


def test_likelihood_variance():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(
        vgp0.likelihood.variance.numpy(),
        cvgp.likelihood.likelihoods[0].variance.numpy(),
        atol=1e-2,
    )
    assert_allclose(
        vgp1.likelihood.variance.numpy(),
        cvgp.likelihood.likelihoods[1].variance.numpy(),
        atol=1e-2,
    )


def test_kernel_variance():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(
        vgp0.kernel.variance.numpy(), cvgp.kernel.kernels[1].kappa.numpy()[0], atol=1.0e-4,
    )
    assert_allclose(
        vgp1.kernel.variance.numpy(), cvgp.kernel.kernels[1].kappa.numpy()[1], atol=1.0e-4,
    )


def test_mean_values():
    vgp0, vgp1, cvgp = _prepare_models()
    assert_allclose(
        vgp0.mean_function.c.numpy(), cvgp.mean_function.meanfunctions[0].c.numpy(), atol=1.0e-4,
    )
    assert_allclose(
        vgp1.mean_function.c.numpy(), cvgp.mean_function.meanfunctions[1].c.numpy(), atol=1.0e-4,
    )


def test_predict_f():
    vgp0, vgp1, cvgp = _prepare_models()

    pred_f0 = vgp0.predict_f(Datum.Xtest)
    pred_fc0 = cvgp.predict_f(Datum.Xtest_augmented0)
    assert_allclose(pred_f0, pred_fc0, atol=1.0e-4)
    pred_f1 = vgp1.predict_f(Datum.Xtest)
    pred_fc1 = cvgp.predict_f(Datum.Xtest_augmented1)
    assert_allclose(pred_f1, pred_fc1, atol=1.0e-4)

    # check predict_f_full_cov
    vgp0.predict_f(Datum.Xtest, full_cov=True)
    cvgp.predict_f(Datum.Xtest_augmented0, full_cov=True)
    vgp1.predict_f(Datum.Xtest, full_cov=True)
    cvgp.predict_f(Datum.Xtest_augmented1, full_cov=True)


def test_predict_y():
    vgp0, vgp1, cvgp = _prepare_models()
    mu1, var1 = vgp0.predict_y(Datum.Xtest)
    c_mu1, c_var1 = cvgp.predict_y(np.hstack([Datum.Xtest, np.zeros((Datum.Xtest.shape[0], 1))]))

    # predict_y returns results for all the likelihoods in multi_likelihood
    assert_allclose(mu1, c_mu1[:, :1], atol=1.0e-4)
    assert_allclose(var1, c_var1[:, :1], atol=1.0e-4)

    mu2, var2 = vgp1.predict_y(Datum.Xtest)
    c_mu2, c_var2 = cvgp.predict_y(np.hstack([Datum.Xtest, np.ones((Datum.Xtest.shape[0], 1))]))

    # predict_y returns results for all the likelihoods in multi_likelihood
    assert_allclose(mu2, c_mu2[:, 1:2], atol=1.0e-4)
    assert_allclose(var2, c_var2[:, 1:2], atol=1.0e-4)


def test_predict_log_density():
    vgp0, vgp1, cvgp = _prepare_models()

    pred_ydensity0 = vgp0.predict_log_density((Datum.Xtest, Datum.Ytest))
    pred_ydensity_c0 = cvgp.predict_log_density((Datum.Xtest_augmented0, Datum.Ytest_augmented0))
    assert_allclose(pred_ydensity0, pred_ydensity_c0, atol=1e-2)
    pred_ydensity1 = vgp1.predict_log_density((Datum.Xtest, Datum.Ytest))
    pred_ydensity_c1 = cvgp.predict_log_density((Datum.Xtest_augmented1, Datum.Ytest_augmented1))
    assert_allclose(pred_ydensity1, pred_ydensity_c1, atol=1e-2)


def test_predict_f_samples():
    vgp0, vgp1, cvgp = _prepare_models()
    # just check predict_f_samples(self) works
    cvgp.predict_f_samples(Datum.X_augmented0, 1)
    cvgp.predict_f_samples(Datum.X_augmented1, 1)
