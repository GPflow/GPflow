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
from dataclasses import dataclass

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose, assert_array_equal, assert_array_less

import gpflow
from gpflow.config import default_float


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: np.ndarray = rng.randn(100, 2)
    Y: np.ndarray = rng.randn(100, 1)
    Z: np.ndarray = rng.randn(10, 2)
    Xs: np.ndarray = rng.randn(10, 2)
    lik = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()


@dataclass(frozen=True)
class DatumSVGP:
    rng: np.random.RandomState = np.random.RandomState(0)
    X = rng.randn(20, 1)
    Y = rng.randn(20, 2)**2
    Z = rng.randn(3, 1)
    qsqrt = (rng.randn(3, 2)**2) * 0.01
    qmean = rng.randn(3, 2)
    lik = gpflow.likelihoods.Exponential()
    data = (X, Y)


default_datum = Datum()
default_datum_svgp = DatumSVGP()


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
    gpflow.models.VGP((default_datum.X, default_datum.Y), default_datum.kernel, default_datum.lik),
    gpflow.models.GPMC((default_datum.X, default_datum.Y), default_datum.kernel, default_datum.lik),
    gpflow.models.SGPMC((default_datum.X, default_datum.Y),
                        default_datum.kernel,
                        default_datum.lik,
                        inducing_variable=default_datum.Z),
    gpflow.models.SGPR((default_datum.X, default_datum.Y), default_datum.kernel, inducing_variable=default_datum.Z),
    gpflow.models.GPR((default_datum.X, default_datum.Y), default_datum.kernel),
    gpflow.models.GPRFITC((default_datum.X, default_datum.Y), default_datum.kernel, inducing_variable=default_datum.Z)
]

_state_less_gp_models = [gpflow.models.SVGP(default_datum.kernel, default_datum.lik, inducing_variable=default_datum.Z)]


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_f(model):
    mf, vf = model.predict_f(default_datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_y(model):
    mf, vf = model.predict_y(default_datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize('model', _state_less_gp_models + _gp_models)
def test_methods_predict_log_density(model):
    rng = Datum().rng
    Ys = rng.randn(10, 1)
    d = model.predict_log_density((default_datum.Xs, Ys))
    assert_array_equal(d.shape, (10, 1))


def test_sgpr_qu():
    rng = Datum().rng
    X, Z = tf.cast(rng.randn(100, 2), default_float()), tf.cast(rng.randn(20, 2), default_float())
    Y = tf.cast(np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * np.random.randn(len(X), 1), default_float())
    model = gpflow.models.SGPR((X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z)

    @tf.function
    def closure():
        return - model.log_marginal_likelihood()

    gpflow.optimizers.Scipy().minimize(closure, variables=model.trainable_variables)

    qu_mean, qu_cov = model.compute_qu()
    f_at_Z_mean, f_at_Z_cov = model.predict_f(model.inducing_variable.Z, full_cov=True)

    np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5)


def test_svgp_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening.
    """
    num_latent = default_datum_svgp.Y.shape[1]
    model_1 = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                                 likelihood=default_datum_svgp.lik,
                                 q_diag=True,
                                 num_latent=num_latent,
                                 inducing_variable=default_datum_svgp.Z,
                                 whiten=True)
    model_2 = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                                 likelihood=default_datum_svgp.lik,
                                 q_diag=False,
                                 num_latent=num_latent,
                                 inducing_variable=default_datum_svgp.Z,
                                 whiten=True)
    model_1.q_sqrt.assign(default_datum_svgp.qsqrt)
    model_1.q_mu.assign(default_datum_svgp.qmean)
    model_2.q_sqrt.assign(np.array([np.diag(default_datum_svgp.qsqrt[:, 0]), np.diag(default_datum_svgp.qsqrt[:, 1])]))
    model_2.q_mu.assign(default_datum_svgp.qmean)
    assert_allclose(model_1.log_likelihood(default_datum_svgp.data), model_2.log_likelihood(default_datum_svgp.data))


def test_svgp_non_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening is not used.
    """
    num_latent = default_datum_svgp.Y.shape[1]
    model_1 = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                                 likelihood=default_datum_svgp.lik,
                                 q_diag=True,
                                 num_latent=num_latent,
                                 inducing_variable=default_datum_svgp.Z,
                                 whiten=False)
    model_2 = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                                 likelihood=default_datum_svgp.lik,
                                 q_diag=False,
                                 num_latent=num_latent,
                                 inducing_variable=default_datum_svgp.Z,
                                 whiten=False)
    model_1.q_sqrt.assign(default_datum_svgp.qsqrt)
    model_1.q_mu.assign(default_datum_svgp.qmean)
    model_2.q_sqrt.assign(np.array([np.diag(default_datum_svgp.qsqrt[:, 0]), np.diag(default_datum_svgp.qsqrt[:, 1])]))
    model_2.q_mu.assign(default_datum_svgp.qmean)
    assert_allclose(model_1.log_likelihood(default_datum_svgp.data), model_2.log_likelihood(default_datum_svgp.data))


def test_svgp_fixing_q_sqrt():
    """
    In response to bug #46, we need to make sure that the q_sqrt matrix can be fixed
    """
    num_latent = default_datum_svgp.Y.shape[1]
    model = gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                               likelihood=default_datum_svgp.lik,
                               q_diag=True,
                               num_latent=num_latent,
                               inducing_variable=default_datum_svgp.Z,
                               whiten=False)
    default_num_trainable_variables = len(model.trainable_variables)
    model.q_sqrt.trainable = False
    assert len(model.trainable_variables) == default_num_trainable_variables - 1


@pytest.mark.parametrize('indices_1, indices_2, num_data1, num_data2, max_iter', [
    [[0, 1], [1, 0], 2, 2, 3],
    [[0, 1], [0, 0], 1, 2, 1],
    [[0, 0], [0, 1], 1, 1, 2],
])
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
        return gpflow.models.SVGP(kernel=gpflow.kernels.SquaredExponential(),
                                  num_data=num_data,
                                  likelihood=gpflow.likelihoods.Gaussian(),
                                  inducing_variable=Z)

    def training_loop(indices, num_data, max_iter):
        model = get_model(num_data)
        opt = tf.optimizers.SGD(learning_rate=.001)
        data = X[indices], Y[indices]
        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                loss = model.log_likelihood(data)
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
    rng = Datum().rng
    X, Y = rng.randn(10, 1), rng.randn(10, 1)
    v_vals = rng.randn(10, 1)

    likelihood = gpflow.likelihoods.StudentT()
    model_1 = gpflow.models.GPMC(data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood)
    model_2 = gpflow.models.SGPMC(data=(X, Y),
                                  kernel=gpflow.kernels.Exponential(),
                                  inducing_variable=X.copy(),
                                  likelihood=likelihood)
    model_1.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_2.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_1.kernel.lengthscale.assign(0.8)
    model_2.kernel.lengthscale.assign(0.8)
    model_1.kernel.variance.assign(4.2)
    model_2.kernel.variance.assign(4.2)

    assert_allclose(model_1.log_likelihood(), model_2.log_likelihood(), rtol=1e-5, atol=1e-5)
