# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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
import tensorflow_probability as tfp

from gpflow import set_trainable
from gpflow.base import AnyNDArray
from gpflow.functions import Linear
from gpflow.inducing_variables import InducingPoints, SeparateIndependentInducingVariables
from gpflow.kernels import SeparateIndependent, SquaredExponential
from gpflow.likelihoods import HeteroskedasticTFPConditional
from gpflow.models import SVGP
from gpflow.optimizers import NaturalGradient

tf.random.set_seed(99012)


class Data:
    rng = np.random.RandomState(123)
    N = 5
    X = np.linspace(0, 1, num=N)[:, None]
    Y = rng.randn(N, 1)
    f_mean = rng.randn(N, 2)
    f_var: AnyNDArray = rng.randn(N, 2) ** 2


def test_analytic_mean_and_var() -> None:
    """
    Test that quadrature computation used in HeteroskedasticTFPConditional
    of the predictive mean and variance is close to the analytical version,
    which can be computed for the special case of N(y | mean=f1, scale=exp(f2)),
    where f1, f2 ~ GP.
    """
    analytic_mean = Data.f_mean[:, [0]]
    analytic_variance = np.exp(Data.f_mean[:, [1]] + Data.f_var[:, [1]]) ** 2 + Data.f_var[:, [0]]

    likelihood = HeteroskedasticTFPConditional()
    y_mean, y_var = likelihood.predict_mean_and_var(Data.X, Data.f_mean, Data.f_var)

    np.testing.assert_allclose(y_mean, analytic_mean)
    np.testing.assert_allclose(y_var, analytic_variance, rtol=1.5e-6)


@pytest.mark.parametrize("A", [0, 1])
def test_heteroskedastic_with_linear_mean_issue_2086(A: float) -> None:

    X = np.linspace(0, 4 * np.pi, 1001)[:, None]
    Y = np.random.normal(np.sin(X), np.exp(np.cos(X)))

    likelihood = HeteroskedasticTFPConditional(
        distribution_class=tfp.distributions.Normal,
        scale_transform=tfp.bijectors.Exp(),
    )
    assert likelihood.latent_dim == 2
    kernel = SeparateIndependent([SquaredExponential(), SquaredExponential()])

    Z = np.linspace(X.min(), X.max(), 20)[:, None]

    inducing_variable = SeparateIndependentInducingVariables([InducingPoints(Z), InducingPoints(Z)])

    mean_f = Linear(A=np.array([[A]]))
    model = SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=inducing_variable,
        num_latent_gps=likelihood.latent_dim,
        mean_function=mean_f,
    )

    data = (X, Y)
    loss_fn = model.training_loss_closure(data)

    set_trainable(model.q_mu, False)
    set_trainable(model.q_sqrt, False)

    variational_vars = [(model.q_mu, model.q_sqrt)]
    natgrad_opt = NaturalGradient(gamma=0.1)

    adam_vars = model.trainable_variables
    adam_opt = tf.optimizers.Adam(0.01)

    @tf.function
    def optimisation_step() -> None:
        natgrad_opt.minimize(loss_fn, variational_vars)
        adam_opt.minimize(loss_fn, adam_vars)

    for epoch in range(1, 51):
        optimisation_step()
