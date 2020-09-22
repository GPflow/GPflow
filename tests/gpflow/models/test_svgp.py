# Copyright 2016-2020 the GPflow authors.
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
from gpflow import set_trainable
from gpflow.config import default_float


@dataclass(frozen=True)
class DatumSVGP:
    rng: np.random.RandomState = np.random.RandomState(0)
    X = rng.randn(20, 1)
    Y = rng.randn(20, 2) ** 2
    Z = rng.randn(3, 1)
    qsqrt = (rng.randn(3, 2) ** 2) * 0.01
    qmean = rng.randn(3, 2)
    lik = gpflow.likelihoods.Exponential()
    data = (X, Y)


default_datum_svgp = DatumSVGP()


def test_svgp_fixing_q_sqrt():
    """
    In response to bug #46, we need to make sure that the q_sqrt matrix can be fixed
    """
    num_latent_gps = default_datum_svgp.Y.shape[1]
    model = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=default_datum_svgp.lik,
        q_diag=True,
        num_latent_gps=num_latent_gps,
        inducing_variable=default_datum_svgp.Z,
        whiten=False,
    )
    default_num_trainable_variables = len(model.trainable_variables)
    set_trainable(model.q_sqrt, False)
    assert len(model.trainable_variables) == default_num_trainable_variables - 1


def test_svgp_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening.
    """
    num_latent_gps = default_datum_svgp.Y.shape[1]
    model_1 = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=default_datum_svgp.lik,
        q_diag=True,
        num_latent_gps=num_latent_gps,
        inducing_variable=default_datum_svgp.Z,
        whiten=True,
    )
    model_2 = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=default_datum_svgp.lik,
        q_diag=False,
        num_latent_gps=num_latent_gps,
        inducing_variable=default_datum_svgp.Z,
        whiten=True,
    )
    model_1.q_sqrt.assign(default_datum_svgp.qsqrt)
    model_1.q_mu.assign(default_datum_svgp.qmean)
    model_2.q_sqrt.assign(
        np.array(
            [np.diag(default_datum_svgp.qsqrt[:, 0]), np.diag(default_datum_svgp.qsqrt[:, 1]),]
        )
    )
    model_2.q_mu.assign(default_datum_svgp.qmean)
    assert_allclose(model_1.elbo(default_datum_svgp.data), model_2.elbo(default_datum_svgp.data))


def test_svgp_non_white():
    """
    Tests that the SVGP bound on the likelihood is the same when using
    with and without diagonals when whitening is not used.
    """
    num_latent_gps = default_datum_svgp.Y.shape[1]
    model_1 = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=default_datum_svgp.lik,
        q_diag=True,
        num_latent_gps=num_latent_gps,
        inducing_variable=default_datum_svgp.Z,
        whiten=False,
    )
    model_2 = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=default_datum_svgp.lik,
        q_diag=False,
        num_latent_gps=num_latent_gps,
        inducing_variable=default_datum_svgp.Z,
        whiten=False,
    )
    model_1.q_sqrt.assign(default_datum_svgp.qsqrt)
    model_1.q_mu.assign(default_datum_svgp.qmean)
    model_2.q_sqrt.assign(
        np.array(
            [np.diag(default_datum_svgp.qsqrt[:, 0]), np.diag(default_datum_svgp.qsqrt[:, 1]),]
        )
    )
    model_2.q_mu.assign(default_datum_svgp.qmean)
    assert_allclose(model_1.elbo(default_datum_svgp.data), model_2.elbo(default_datum_svgp.data))


def _check_models_close(m1, m2, tolerance=1e-2):
    m1_params = {p.name: p for p in list(m1.trainable_parameters)}
    m2_params = {p.name: p for p in list(m2.trainable_parameters)}
    if set(m2_params.keys()) != set(m2_params.keys()):
        return False
    for key in m1_params:
        p1 = m1_params[key]
        p2 = m2_params[key]
        if not np.allclose(p1.numpy(), p2.numpy(), rtol=tolerance, atol=tolerance):
            return False
    return True


@pytest.mark.parametrize(
    "indices_1, indices_2, num_data1, num_data2, max_iter",
    [[[0, 1], [1, 0], 2, 2, 3], [[0, 1], [0, 0], 1, 2, 1], [[0, 0], [0, 1], 1, 1, 2],],
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
    X, Y = np.atleast_2d(np.array([0.0, 1.0])).T, np.atleast_2d(np.array([-1.0, 3.0])).T
    Z = np.atleast_2d(np.array([0.5]))

    def get_model(num_data):
        return gpflow.models.SVGP(
            kernel=gpflow.kernels.SquaredExponential(),
            num_data=num_data,
            likelihood=gpflow.likelihoods.Gaussian(),
            inducing_variable=Z,
        )

    def training_loop(indices, num_data, max_iter):
        model = get_model(num_data)
        opt = tf.optimizers.SGD(learning_rate=0.001)
        data = X[indices], Y[indices]
        for _ in range(max_iter):
            with tf.GradientTape() as tape:
                loss = model.training_loss(data)
            grads = tape.gradient(loss, model.trainable_variables)
            opt.apply_gradients(zip(grads, model.trainable_variables))
        return model

    model_1 = training_loop(indices_1, num_data=num_data1, max_iter=max_iter)
    model_2 = training_loop(indices_2, num_data=num_data2, max_iter=max_iter)
    assert _check_models_close(model_1, model_2)
