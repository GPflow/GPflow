# Copyright 2019 the GPflow authors.
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

from typing import Optional

import numpy as np
import pytest

import gpflow
from gpflow.utilities.ops import pca_reduce


class Data:
    rng = np.random.RandomState(999)
    N = 20
    D = 5
    Y = rng.randn(N, D)
    Q = 2
    M = 10
    X = rng.randn(N, Q)


@pytest.mark.parametrize(
    "kernel",
    [
        None,  # default kernel: SquaredExponential
        gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential()),
    ],
)
def test_gplvm_with_kernels(kernel: Optional[gpflow.kernels.Kernel]) -> None:
    m = gpflow.models.GPLVM(Data.Y, Data.Q, kernel=kernel)
    lml_initial = m.log_marginal_likelihood()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=2))
    assert m.log_marginal_likelihood() > lml_initial


def test_bayesian_gplvm_1d() -> None:
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    m = gpflow.models.BayesianGPLVM(
        Data.Y,
        np.zeros((Data.N, Q)),
        np.ones((Data.N, Q)),
        kernel,
        inducing_variable=inducing_variable,
    )
    assert m.inducing_variable.num_inducing == Data.M

    elbo_initial = m.elbo()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=2))
    assert m.elbo() > elbo_initial


def test_bayesian_gplvm_2d() -> None:
    Q = 2  # latent dimensions
    X_data_mean = pca_reduce(Data.Y, Q)
    kernel = gpflow.kernels.SquaredExponential()

    m = gpflow.models.BayesianGPLVM(
        Data.Y, X_data_mean, np.ones((Data.N, Q)), kernel, num_inducing_variables=Data.M
    )

    elbo_initial = m.elbo()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=2))
    assert m.elbo() > elbo_initial

    # test prediction
    Xtest = Data.rng.randn(10, Q)
    mu_f, var_f = m.predict_f(Xtest)
    mu_fFull, var_fFull = m.predict_f(Xtest, full_cov=True)
    np.testing.assert_allclose(mu_fFull, mu_f)

    for i in range(Data.D):
        np.testing.assert_allclose(var_f[:, i], np.diag(var_fFull[i, :, :]))


def test_gplvm_constructor_checks() -> None:
    with pytest.raises(ValueError):
        assert Data.X.shape[1] == Data.Q
        latents_wrong_shape = Data.X[:, : Data.Q - 1]
        gpflow.models.GPLVM(Data.Y, Data.Q, X_data_mean=latents_wrong_shape)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, : Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, : Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q, X_data_mean=Data.X)


def test_bayesian_gplvm_constructor_check() -> None:
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    with pytest.raises(ValueError):
        gpflow.models.BayesianGPLVM(
            Data.Y,
            np.zeros((Data.N, Q)),
            np.ones((Data.N, Q)),
            kernel,
            inducing_variable=inducing_variable,
            num_inducing_variables=len(inducing_variable),
        )
