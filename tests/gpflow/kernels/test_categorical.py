# Copyright 2026 the GPflow authors.
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
from gpflow.kernels.categorical import Categorical


def test_categorical_kernel_respects_active_dims_ard() -> None:
    rng = np.random.RandomState(0)
    X = np.hstack([rng.rand(12, 2), rng.choice([0, 1, 2], size=(12, 1))])
    X2 = np.hstack([rng.rand(5, 2), rng.choice([0, 1, 2], size=(5, 1))])

    non_cat = gpflow.kernels.RBF(lengthscales=[0.2, 0.5], active_dims=[0, 1])
    cat = gpflow.kernels.RBF(lengthscales=0.3, active_dims=[2])
    kernel = Categorical(
        non_categorical_kernel=non_cat,
        categorical_kernel=cat,
        num_labels=3,
    )

    X_lat = kernel._concat_inputs_with_latents(X)
    X2_lat = kernel._concat_inputs_with_latents(X2)

    assert_allclose(kernel(X), non_cat(X_lat) * cat(X_lat))
    assert_allclose(kernel(X, X2), non_cat(X_lat, X2_lat) * cat(X_lat, X2_lat))
    assert_allclose(
        kernel(X, full_cov=False),
        non_cat(X_lat, full_cov=False) * cat(X_lat, full_cov=False),
    )


def test_categorical_kernel_respects_active_dims_nonstationary_diag() -> None:
    rng = np.random.RandomState(1)
    X = np.hstack([rng.rand(8, 2), rng.choice([0, 1, 2], size=(8, 1))])

    non_cat = gpflow.kernels.Linear(variance=1.5, active_dims=[0, 1])
    cat = gpflow.kernels.RBF(lengthscales=0.4, active_dims=[2])
    kernel = Categorical(
        non_categorical_kernel=non_cat,
        categorical_kernel=cat,
        num_labels=3,
    )

    X_lat = kernel._concat_inputs_with_latents(X)

    assert_allclose(kernel(X), non_cat(X_lat) * cat(X_lat))
    assert_allclose(
        kernel(X, full_cov=False),
        non_cat(X_lat, full_cov=False) * cat(X_lat, full_cov=False),
    )


def test_categorical_kernel_gpr_optimization_with_ard() -> None:
    rng = np.random.RandomState(2)
    X = np.hstack([rng.rand(15, 2), rng.choice([0, 1, 2], size=(15, 1))])
    Y = rng.rand(15, 1)

    kernel = Categorical(
        non_categorical_kernel=gpflow.kernels.RBF(lengthscales=[0.1, 0.1], active_dims=[0, 1]),
        categorical_kernel=gpflow.kernels.RBF(lengthscales=0.1, active_dims=[2]),
        num_labels=3,
    )
    model = gpflow.models.GPR(data=(X, Y), kernel=kernel)
    opt = gpflow.optimizers.Scipy()
    res = opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=5))
    assert np.isfinite(res.fun)
