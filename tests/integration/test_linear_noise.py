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

from typing import Callable

import numpy as np
import pytest

import gpflow
from gpflow import default_float
from gpflow.base import AnyNDArray, RegressionData
from gpflow.config import default_float
from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.functions import Linear
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR, GPRFITC, SGPR, SVGP, VGP, GPModel, training_loss_closure
from gpflow.models.util import InducingPointsLike


class Datum:
    cs = ShapeChecker().check_shape

    rng = np.random.default_rng(20220630)
    n = 100
    X: AnyNDArray = cs(rng.random((n, 1), dtype=default_float()), "[N, 1]")
    noise_slope = -0.7
    noise_offset = 0.7
    noise = cs(
        (noise_slope * X + noise_offset) * rng.standard_normal((n, 1), dtype=default_float()),
        "[N, 1]",
    )
    Y = cs(np.sin(5 * X) + noise, "[N, 1]")
    data = X, Y


def create_kernel() -> Kernel:
    return gpflow.kernels.RBF(lengthscales=0.2)


def create_inducing() -> InducingPointsLike:
    Z = np.linspace(0.0, 1.0, 10)[:, None]
    iv = gpflow.inducing_variables.InducingPoints(Z)
    gpflow.set_trainable(iv.Z, False)
    return iv


def create_linear_noise() -> Gaussian:
    return Gaussian(scale=Linear())


def gpr(data: RegressionData) -> GPModel:
    return GPR(
        data,
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
    )


def vgp(data: RegressionData) -> GPModel:
    return VGP(
        data,
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
    )


def sgpr(data: RegressionData) -> GPModel:
    return SGPR(
        data,
        kernel=create_kernel(),
        inducing_variable=create_inducing(),
        likelihood=create_linear_noise(),
    )


def gprfitc(data: RegressionData) -> GPModel:
    return GPRFITC(
        data,
        kernel=create_kernel(),
        inducing_variable=create_inducing(),
        likelihood=create_linear_noise(),
    )


def svgp(data: RegressionData) -> GPModel:
    return SVGP(
        kernel=create_kernel(),
        likelihood=create_linear_noise(),
        inducing_variable=create_inducing(),
    )


CREATE_MODELS = (
    gpr,
    vgp,
    sgpr,
    gprfitc,
    svgp,
)


@pytest.mark.parametrize("create_model", CREATE_MODELS)
def test_infer_noise(create_model: Callable[[RegressionData], GPModel]) -> None:
    model = create_model(Datum.data)
    gpflow.optimizers.Scipy().minimize(
        training_loss_closure(model, Datum.data),
        variables=model.trainable_variables,
    )

    noise_scale = model.likelihood.scale
    np.testing.assert_allclose(Datum.noise_slope, noise_scale.A, atol=0.1)
    np.testing.assert_allclose(Datum.noise_offset, noise_scale.b, atol=0.1)
