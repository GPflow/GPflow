# Copyright 2020 the GPflow authors.
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
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.likelihoods import Gaussian, StudentT, SwitchedLikelihood


@pytest.mark.parametrize("Y_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("F_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("Fvar_list", [[tf.exp(tf.random.normal((i, 2))) for i in range(3, 6)]])
@pytest.mark.parametrize("Y_label", [[tf.ones((i, 2)) * (i - 3.0) for i in range(3, 6)]])
def test_switched_likelihood_log_prob(Y_list, F_list, Fvar_list, Y_label):
    """
    SwitchedLikelihood is separately tested here.
    Here, we make sure the partition-stitch works fine.
    """
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.log_prob(F_sw, Y_sw)
    results = [lik.log_prob(f, y) for lik, y, f in zip(likelihoods, Y_list, F_list)]

    assert_allclose(switched_results, np.concatenate(results)[Y_perm])


@pytest.mark.parametrize("Y_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("F_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("Fvar_list", [[tf.exp(tf.random.normal((i, 2))) for i in range(3, 6)]])
@pytest.mark.parametrize("Y_label", [[tf.ones((i, 2)) * (i - 3.0) for i in range(3, 6)]])
def test_switched_likelihood_predict_log_density(Y_list, F_list, Fvar_list, Y_label):
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    Fvar_sw = np.concatenate(Fvar_list)[Y_perm, :]

    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.predict_log_density(F_sw, Fvar_sw, Y_sw)
    # likelihood
    results = [
        lik.predict_log_density(f, fvar, y)
        for lik, y, f, fvar in zip(likelihoods, Y_list, F_list, Fvar_list)
    ]
    assert_allclose(switched_results, np.concatenate(results)[Y_perm])


@pytest.mark.parametrize("Y_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("F_list", [[tf.random.normal((i, 2)) for i in range(3, 6)]])
@pytest.mark.parametrize("Fvar_list", [[tf.exp(tf.random.normal((i, 2))) for i in range(3, 6)]])
@pytest.mark.parametrize("Y_label", [[tf.ones((i, 2)) * (i - 3.0) for i in range(3, 6)]])
def test_switched_likelihood_variational_expectations(Y_list, F_list, Fvar_list, Y_label):
    Y_perm = list(range(3 + 4 + 5))
    np.random.shuffle(Y_perm)
    # shuffle the original data
    Y_sw = np.hstack([np.concatenate(Y_list), np.concatenate(Y_label)])[Y_perm, :3]
    F_sw = np.concatenate(F_list)[Y_perm, :]
    Fvar_sw = np.concatenate(Fvar_list)[Y_perm, :]

    likelihoods = [Gaussian()] * 3
    for lik in likelihoods:
        lik.variance = np.exp(np.random.randn(1)).squeeze().astype(np.float32)
    switched_likelihood = SwitchedLikelihood(likelihoods)

    switched_results = switched_likelihood.variational_expectations(F_sw, Fvar_sw, Y_sw)
    results = [
        lik.variational_expectations(f, fvar, y)
        for lik, y, f, fvar in zip(likelihoods, Y_list, F_list, Fvar_list)
    ]
    assert_allclose(switched_results, np.concatenate(results)[Y_perm])


def test_switched_likelihood_with_vgp():
    """
    Reproduces the bug in https://github.com/GPflow/GPflow/issues/951
    """
    X = np.random.randn(12 + 15, 1)
    Y = np.random.randn(12 + 15, 1)
    idx = np.array([0] * 12 + [1] * 15)
    Y_aug = np.c_[Y, idx]
    assert Y_aug.shape == (12 + 15, 2)

    kernel = gpflow.kernels.Matern32()
    likelihood = gpflow.likelihoods.SwitchedLikelihood([StudentT(), StudentT()])
    model = gpflow.models.VGP((X, Y_aug), kernel=kernel, likelihood=likelihood)
    # without bugfix, optimization errors out
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=1))


@pytest.mark.parametrize("num_latent_gps", [1, 2])
def test_switched_likelihood_regression_valid_num_latent_gps(num_latent_gps):
    """
    A Regression test when using Switched likelihood: the number of latent
    functions in a GP model must be equal to the number of columns in Y minus
    one. The final column of Y is used to index the switch. If the number of
    latent functions does not match, an exception will be raised.
    """
    x = np.random.randn(100, 1)
    y = np.hstack((np.random.randn(100, 1), np.random.randint(0, 3, (100, 1))))
    data = x, y

    Z = InducingPoints(np.random.randn(num_latent_gps, 1))
    likelihoods = [StudentT()] * 3
    switched_likelihood = SwitchedLikelihood(likelihoods)
    m = gpflow.models.SVGP(
        kernel=gpflow.kernels.Matern12(),
        inducing_variable=Z,
        likelihood=switched_likelihood,
        num_latent_gps=num_latent_gps,
    )
    if num_latent_gps == 1:
        _ = m.training_loss(data)
    else:
        with pytest.raises(tf.errors.InvalidArgumentError):
            _ = m.training_loss(data)
