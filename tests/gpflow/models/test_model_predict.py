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

from typing import Any, Dict, Optional, Type

import numpy as np
import pytest

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import Kernel, Matern32
from gpflow.models.util import InducingPointsLike

rng = np.random.RandomState(0)


class ModelSetup:
    def __init__(
        self,
        model_class: Type[gpflow.models.GPModel],
        kernel: Kernel = Matern32(),
        likelihood: gpflow.likelihoods.Likelihood = gpflow.likelihoods.Gaussian(),
        whiten: Optional[bool] = None,
        q_diag: Optional[bool] = None,
        requires_inducing_variables: bool = True,
        requires_data: bool = False,
        requires_likelihood: bool = True,
    ) -> None:

        self.model_class = model_class
        self.kernel = kernel
        self.likelihood = likelihood
        self.whiten = whiten
        self.q_diag = q_diag
        self.requires_inducing_variables = requires_inducing_variables
        self.requires_data = requires_data
        self.requires_likelihood = requires_likelihood

    def get_model(
        self,
        Z: InducingPointsLike,
        num_latent_gps: int,
        data: Optional[gpflow.base.RegressionData] = None,
    ) -> gpflow.models.GPModel:
        params: Dict[str, Any] = dict(kernel=self.kernel, num_latent_gps=num_latent_gps)

        if self.whiten is not None and self.q_diag is not None:
            params.update(inducing_variable=Z, whiten=self.whiten, q_diag=self.q_diag)

        if self.requires_inducing_variables:
            params.update(dict(inducing_variable=Z))

        if self.requires_data:
            params.update(dict(data=data))

        if self.requires_likelihood:
            params.update(dict(likelihood=self.likelihood))

        return self.model_class(**params)

    def __repr__(self) -> str:
        return f"ModelSetup({self.model_class.__name__}, {self.whiten}, {self.q_diag})"


model_setups = [
    ModelSetup(model_class=gpflow.models.SVGP, whiten=False, q_diag=True),
    ModelSetup(model_class=gpflow.models.SVGP, whiten=True, q_diag=False),
    ModelSetup(model_class=gpflow.models.SVGP, whiten=True, q_diag=True),
    ModelSetup(model_class=gpflow.models.SVGP, whiten=False, q_diag=False),
    ModelSetup(
        model_class=gpflow.models.SGPR,  # type: ignore[type-abstract, misc]  # mypy thinks SPGR is abstract.
        requires_data=True,
        requires_likelihood=False,
    ),
    ModelSetup(
        model_class=gpflow.models.VGP,
        requires_inducing_variables=False,
        requires_data=True,
    ),
    #     ModelSetup(model_class=gpflow.models.GPRF),
    ModelSetup(
        model_class=gpflow.models.GPMC,
        requires_data=True,
        requires_inducing_variables=False,
    ),
    ModelSetup(
        model_class=gpflow.models.SGPMC,
        requires_data=True,
        requires_inducing_variables=True,
    ),
]


@pytest.mark.parametrize("Ntrain, Ntest, D", [[100, 10, 2]])
def test_gaussian_mean_and_variance(Ntrain: int, Ntest: int, D: int) -> None:
    data = rng.randn(Ntrain, D), rng.randn(Ntrain, 1)
    Xtest, _ = rng.randn(Ntest, D), rng.randn(Ntest, 1)
    kernel = Matern32() + gpflow.kernels.White()
    model_gp = gpflow.models.GPR(data, kernel=kernel)

    mu_f, var_f = model_gp.predict_f(Xtest)
    mu_y, var_y = model_gp.predict_y(Xtest)

    assert np.allclose(mu_f, mu_y)
    assert np.allclose(var_f, var_y - 1.0)


@pytest.mark.parametrize("Ntrain, Ntest, D", [[100, 10, 2]])
def test_gaussian_log_density(Ntrain: int, Ntest: int, D: int) -> None:
    data = rng.randn(Ntrain, D), rng.randn(Ntrain, 1)
    Xtest, Ytest = rng.randn(Ntest, D), rng.randn(Ntest, 1)
    kernel = Matern32() + gpflow.kernels.White()
    model_gp = gpflow.models.GPR(data, kernel=kernel)

    mu_y, var_y = model_gp.predict_y(Xtest)
    data = Xtest, Ytest
    log_density = model_gp.predict_log_density(data)
    log_density_hand = np.squeeze(
        -0.5 * np.log(2 * np.pi) - 0.5 * np.log(var_y) - 0.5 * np.square(mu_y - Ytest) / var_y,
        axis=-1,
    )

    assert np.allclose(log_density_hand, log_density)


@pytest.mark.parametrize("input_dim, output_dim, N, Ntest", [[3, 2, 20, 30]])
def test_gaussian_full_cov(input_dim: int, output_dim: int, N: int, Ntest: int) -> None:
    covar_shape = (output_dim, Ntest, Ntest)
    X, Y = rng.randn(N, input_dim), rng.randn(N, output_dim)
    Xtest = rng.randn(Ntest, input_dim)
    kernel = Matern32()
    model_gp = gpflow.models.GPR((X, Y), kernel=kernel)

    mu1, var = model_gp.predict_f(Xtest, full_cov=False)
    mu2, covar = model_gp.predict_f(Xtest, full_cov=True)

    assert np.allclose(mu1, mu2, atol=1.0e-10)
    assert covar.shape == covar_shape
    assert var.shape == (Ntest, output_dim)
    for i in range(output_dim):
        assert np.allclose(var[:, i], np.diag(covar[i, :, :]))


@pytest.mark.parametrize("input_dim, output_dim, N, Ntest, M, num_samples", [[3, 2, 20, 30, 5, 5]])
def test_gaussian_full_cov_samples(
    input_dim: int, output_dim: int, N: int, Ntest: int, M: int, num_samples: int
) -> None:
    samples_shape = (num_samples, Ntest, output_dim)
    X, Y, _ = rng.randn(N, input_dim), rng.randn(N, output_dim), rng.randn(M, input_dim)
    Xtest = rng.randn(Ntest, input_dim)
    kernel = Matern32()
    model_gp = gpflow.models.GPR((X, Y), kernel=kernel)

    samples = model_gp.predict_f_samples(Xtest, num_samples)
    assert samples.shape == samples_shape

    samples = model_gp.predict_f_samples(Xtest, num_samples, full_cov=False)
    assert samples.shape == samples_shape


@pytest.mark.parametrize("model_setup", model_setups)
@pytest.mark.parametrize("input_dim", [3])
@pytest.mark.parametrize("output_dim", [2])
@pytest.mark.parametrize("N", [20])
@pytest.mark.parametrize("Ntest", [30])
@pytest.mark.parametrize("M", [5])
def test_other_models_full_cov(
    model_setup: ModelSetup, input_dim: int, output_dim: int, N: int, Ntest: int, M: int
) -> None:
    covar_shape = (output_dim, Ntest, Ntest)
    X, Y = rng.randn(N, input_dim), rng.randn(N, output_dim)
    Z = InducingPoints(rng.randn(M, input_dim))
    Xtest = rng.randn(Ntest, input_dim)
    model_gp = model_setup.get_model(Z, num_latent_gps=output_dim, data=(X, Y))

    mu1, var = model_gp.predict_f(Xtest, full_cov=False)
    mu2, covar = model_gp.predict_f(Xtest, full_cov=True)

    assert np.allclose(mu1, mu2, atol=1.0e-10)
    assert covar.shape == covar_shape
    assert var.shape == (Ntest, output_dim)
    for i in range(output_dim):
        assert np.allclose(var[:, i], np.diag(covar[i, :, :]))


@pytest.mark.parametrize("model_setup", model_setups)
@pytest.mark.parametrize("input_dim", [3])
@pytest.mark.parametrize("output_dim", [2])
@pytest.mark.parametrize("N", [20])
@pytest.mark.parametrize("Ntest", [30])
@pytest.mark.parametrize("M", [5])
@pytest.mark.parametrize("num_samples", [5])
def test_other_models_full_cov_samples(
    model_setup: ModelSetup,
    input_dim: int,
    output_dim: int,
    N: int,
    Ntest: int,
    M: int,
    num_samples: int,
) -> None:
    samples_shape = (num_samples, Ntest, output_dim)
    X, Y, Z = rng.randn(N, input_dim), rng.randn(N, output_dim), rng.randn(M, input_dim)
    Xtest = rng.randn(Ntest, input_dim)
    model_gp = model_setup.get_model(Z, num_latent_gps=output_dim, data=(X, Y))

    samples = model_gp.predict_f_samples(Xtest, num_samples)
    assert samples.shape == samples_shape
