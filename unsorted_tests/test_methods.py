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
import pytest

import gpflow
import numpy as np
import tensorflow as tf

from numpy.testing import assert_array_equal, assert_array_less, assert_allclose

from gpflow.utilities.defaults import default_float

rng = np.random.RandomState(0)


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------

class Datum:
    X = rng.randn(100, 2)
    Y = rng.randn(100, 1)
    Z = rng.randn(10, 2)
    Xs = rng.randn(10, 2)
    lik = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()


class DatumSVGP:
    X = rng.randn(20, 1)
    Y = rng.randn(20, 2) ** 2
    Z = rng.randn(3, 1)
    qsqrt, qmean = rng.randn(2, 3, 2)
    qsqrt = (qsqrt ** 2) * 0.01
    lik = gpflow.likelihoods.Exponential()


def _check_models_close(m1, m2, tolerance=1e-2):
    m1_params = {p.name: p for p in list(m1.trainable_parameters)}
    m2_params = {p.name: p for p in list(m2.trainable_parameters)}
    if set(m2_params.keys()) != set(m2_params.keys()):
        return False
    for key in m1_params:
        p1 = m1_params[key]
        p2 = m2_params[key]
        if not np.allclose(p1.read_value(), p2.read_value(), rtol=tolerance, atol=tolerance):
            return False
    return True


_gp_models = [
    gpflow.models.VGP(Datum.X, Datum.Y, Datum.kernel, Datum.lik),
    gpflow.models.GPMC(Datum.X, Datum.Y, Datum.kernel, Datum.lik),
    gpflow.models.SGPMC(Datum.X, Datum.Y, Datum.kernel, Datum.lik, features=Datum.Z),
    gpflow.models.SGPR(Datum.X, Datum.Y, Datum.kernel, features=Datum.Z),
    gpflow.models.GPR(Datum.X, Datum.Y, Datum.kernel),
    gpflow.models.GPRFITC(Datum.X, Datum.Y, Datum.kernel, features=Datum.Z)
]

_state_less_gp_models = [
    gpflow.models.SVGP(Datum.kernel, Datum.lik, feature=Datum.Z)
]


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_f(model):
    mf, vf = model.predict_f(Datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_y(model):
    mf, vf = model.predict_y(Datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_log_density(model):
    Ys = rng.randn(10, 1)
    d = model.predict_log_density(Datum.Xs, Ys)
    assert_array_equal(d.shape, (10, 1))


def test_sgpr_qu():
    X, Z = rng.randn(100, 2), rng.randn(20, 2)
    Y = np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * np.random.randn(len(X), 1)
    model = gpflow.models.SGPR(X, Y, gpflow.kernels.RBF(), features=Z)

    # @tf.function
    def closure():
        return - model.log_likelihood()

    gpflow.optimizers.Scipy().minimize(closure, variables=model.trainable_variables)

    q_mu = model.mean_function(model.feature.Z.read_value())
    q_var = model.kernel(model.feature.Z.read_value(), full=True)
    model_qu = [q_mu, tf.reshape(q_var, (1, 20, 20))]

    for v1, v2 in zip(model_qu, model.predict_f(model.feature.Z.read_value(), full_cov=True)):
        np.testing.assert_allclose(v1, v2, rtol=0, atol=1e-7)


def test_svgp_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening.
    """
    num_latent = DatumSVGP.Y.shape[1]
    model_1 = gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), likelihood=DatumSVGP.lik, q_diag=True,
                                 num_latent=num_latent, feature=DatumSVGP.Z, whiten=True)
    model_2 = gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), likelihood=DatumSVGP.lik,
                                 q_diag=False,
                                 num_latent=num_latent, feature=DatumSVGP.Z, whiten=True)
    model_1.q_sqrt.assign(DatumSVGP.qsqrt)
    model_1.q_mu.assign(DatumSVGP.qmean)
    model_2.q_sqrt.assign(np.array([np.diag(DatumSVGP.qsqrt[:, 0]),
                                    np.diag(DatumSVGP.qsqrt[:, 1])]))
    model_2.q_mu.assign(DatumSVGP.qmean)
    assert_allclose(model_1.log_likelihood(DatumSVGP.X, DatumSVGP.Y),
                    model_2.log_likelihood(DatumSVGP.X, DatumSVGP.Y))


def test_svgp_non_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening is not used.
    """
    num_latent = DatumSVGP.Y.shape[1]
    model_1 = gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), likelihood=DatumSVGP.lik, q_diag=True,
                                 num_latent=num_latent, feature=DatumSVGP.Z, whiten=False)
    model_2 = gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), likelihood=DatumSVGP.lik,
                                 q_diag=False,
                                 num_latent=num_latent, feature=DatumSVGP.Z, whiten=False)
    model_1.q_sqrt.assign(DatumSVGP.qsqrt)
    model_1.q_mu.assign(DatumSVGP.qmean)
    model_2.q_sqrt.assign(np.array([np.diag(DatumSVGP.qsqrt[:, 0]),
                                    np.diag(DatumSVGP.qsqrt[:, 1])]))
    model_2.q_mu.assign(DatumSVGP.qmean)
    assert_allclose(model_1.log_likelihood(DatumSVGP.X, DatumSVGP.Y),
                    model_2.log_likelihood(DatumSVGP.X, DatumSVGP.Y))


def test_svgp_fixing_q_sqrt():
    """
    In response to bug #46, we need to make sure that the q_sqrt matrix can be fixed
    """
    num_latent = DatumSVGP.Y.shape[1]
    model = gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), likelihood=DatumSVGP.lik, q_diag=True,
                               num_latent=num_latent, feature=DatumSVGP.Z, whiten=False)
    default_num_trainable_variables = len(model.trainable_variables)
    model.q_sqrt.trainable = False
    assert len(model.trainable_variables) == default_num_trainable_variables - 1


@pytest.mark.parametrize(
    'indices_1, indices_2, num_data1, num_data2, max_iter', [
        [[0, 1], [1, 0], 2, 2, 3],
        [[0, 1], [0, 0], 1, 2, 1],
        [[0, 0], [0, 1], 1, 1, 2],
    ]
)
def test_stochastic_gradients(indices_1, indices_2, num_data1, num_data2, max_iter):
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
    X, Y = np.atleast_2d(np.array([0., 1.])).T, np.atleast_2d(np.array([-1., 3.])).T
    Z = np.atleast_2d(np.array([0.5]))

    def get_model(num_data):
        return gpflow.models.SVGP(kernel=gpflow.kernels.RBF(), num_data=num_data,
                                  likelihood=gpflow.likelihoods.Gaussian(), feature=Z)

    def training_loop(indices, num_data, max_iter):
        model = get_model(num_data)
        opt = tf.optimizers.SGD(learning_rate=.001)
        Xnew, Ynew = X[indices], Y[indices]
        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                loss = model.log_likelihood(Xnew, Ynew)
                grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        return model

    model_1 = training_loop(indices_1, num_data=num_data1, max_iter=max_iter)
    model_2 = training_loop(indices_2, num_data=num_data2, max_iter=max_iter)
    assert _check_models_close(model_1, model_2)


def test_sparse_mcmc_likelihoods_and_gradients():
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    X, Y = rng.randn(10, 1), rng.randn(10, 1)
    v_vals = rng.randn(10, 1)

    likelihood = gpflow.likelihoods.StudentT()
    model_1 = gpflow.models.GPMC(X=X, Y=Y, kernel=gpflow.kernels.Exponential(),
                                 likelihood=likelihood)
    model_2 = gpflow.models.SGPMC(X=X, Y=Y, kernel=gpflow.kernels.Exponential(), features=X.copy(),
                                  likelihood=likelihood)
    model_1.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_2.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_1.kernel.lengthscale = .8
    model_2.kernel.lengthscale = .8
    model_1.kernel.variance = 4.2
    model_2.kernel.variance = 4.2

    assert_allclose(model_1.log_likelihood(), model_2.log_likelihood(), rtol=1e-5, atol=1e-5)
