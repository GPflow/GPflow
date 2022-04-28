# Copyright 2016 the GPflow authors
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
from typing import List

import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_array_less

import gpflow
from gpflow.base import AnyNDArray

# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


@dataclass(frozen=True)
class Datum:
    rng: np.random.RandomState = np.random.RandomState(0)
    X: AnyNDArray = rng.randn(100, 2)
    Y: AnyNDArray = rng.randn(100, 1)
    Z: AnyNDArray = rng.randn(10, 2)
    Xs: AnyNDArray = rng.randn(10, 2)
    lik = gpflow.likelihoods.Gaussian()
    kernel = gpflow.kernels.Matern32()


default_datum = Datum()


_gp_models: List[gpflow.models.GPModel] = [
    gpflow.models.VGP((default_datum.X, default_datum.Y), default_datum.kernel, default_datum.lik),
    gpflow.models.GPMC((default_datum.X, default_datum.Y), default_datum.kernel, default_datum.lik),
    gpflow.models.SGPMC(
        (default_datum.X, default_datum.Y),
        default_datum.kernel,
        default_datum.lik,
        inducing_variable=default_datum.Z,
    ),
    gpflow.models.SGPR(
        (default_datum.X, default_datum.Y),
        default_datum.kernel,
        inducing_variable=default_datum.Z,
    ),
    gpflow.models.GPR((default_datum.X, default_datum.Y), default_datum.kernel),
    gpflow.models.GPRFITC(
        (default_datum.X, default_datum.Y),
        default_datum.kernel,
        inducing_variable=default_datum.Z,
    ),
]

_state_less_gp_models: List[gpflow.models.GPModel] = [
    gpflow.models.SVGP(default_datum.kernel, default_datum.lik, inducing_variable=default_datum.Z)
]


@pytest.mark.parametrize("model", _state_less_gp_models + _gp_models)
def test_methods_predict_f(model: gpflow.models.GPModel) -> None:
    mf, vf = model.predict_f(default_datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize("model", _state_less_gp_models + _gp_models)
def test_methods_predict_y(model: gpflow.models.GPModel) -> None:
    mf, vf = model.predict_y(default_datum.Xs)
    assert_array_equal(mf.shape, vf.shape)
    assert_array_equal(mf.shape, (10, 1))
    assert_array_less(np.full_like(vf, -1e-6), vf)


@pytest.mark.parametrize("model", _state_less_gp_models + _gp_models)
def test_methods_predict_log_density(model: gpflow.models.GPModel) -> None:
    rng = Datum().rng
    Ys = rng.randn(10, 1)
    d = model.predict_log_density((default_datum.Xs, Ys))
    assert_array_equal(d.shape, (10,))
