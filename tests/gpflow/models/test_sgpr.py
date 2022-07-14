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
import tensorflow as tf

import gpflow
from gpflow.base import AnyNDArray
from gpflow.utilities import to_default_float


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: AnyNDArray = rng.randn(100, 2)
    Y: AnyNDArray = rng.randn(100, 1)
    Z: AnyNDArray = rng.randn(20, 2)


def test_sgpr_qu() -> None:
    rng = np.random.RandomState(1)
    X = to_default_float(Datum.X)
    Z = to_default_float(Datum.Z)
    Y = to_default_float(np.sin(X @ np.array([[-1.4], [0.5]])) + 0.5 * rng.randn(len(X), 1))
    model = gpflow.models.SGPR(
        (X, Y), kernel=gpflow.kernels.SquaredExponential(), inducing_variable=Z
    )

    gpflow.optimizers.Scipy().minimize(model.training_loss, variables=model.trainable_variables)

    qu_mean, qu_cov = model.compute_qu()
    f_at_Z_mean, f_at_Z_cov = model.predict_f(model.inducing_variable.Z, full_cov=True)

    np.testing.assert_allclose(qu_mean, f_at_Z_mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(tf.reshape(qu_cov, (1, 20, 20)), f_at_Z_cov, rtol=1e-5, atol=1e-5)


def test_sgpr_svgp_qu_equivivalent() -> None:
    rng = np.random.RandomState(2)
    X = to_default_float(Datum.X)
    Y = to_default_float(Datum.Y)
    Z = to_default_float(Datum.Z)
    kernel = gpflow.kernels.SquaredExponential()
    linear = gpflow.functions.Linear(A=[[0.1], [0.2]], b=0.3)
    likelihood = gpflow.likelihoods.Gaussian(scale=linear)
    sgpr = gpflow.models.SGPR(
        (X, Y),
        kernel=kernel,
        inducing_variable=Z,
        likelihood=likelihood,
    )

    q_mu, q_cov = sgpr.compute_qu()
    q_sqrt = tf.linalg.cholesky(q_cov)
    q_sqrt = q_sqrt[None, :, :]
    svgp = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=False,
    )

    Xnew = to_default_float(rng.randn(100, 2))
    np.testing.assert_allclose(
        sgpr.predict_f(Xnew),
        svgp.predict_f(Xnew),
        atol=1e-4,
    )
