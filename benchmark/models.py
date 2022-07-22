# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
"""
Concrete code for creating models.
"""
import numpy as np
from scipy.stats.qmc import Halton  # pylint: disable=no-name-in-module

import benchmark.dataset_api as ds
from benchmark.dataset_api import XYData
from benchmark.model_api import REGRESSION, SPARSE, VARIATIONAL, make_model_factory
from gpflow.inducing_variables import InducingPoints
from gpflow.kernels import RBF, Kernel
from gpflow.likelihoods import Gaussian
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel
from gpflow.models.util import InducingPointsLike


def create_rbf(data: XYData, rng: np.random.Generator) -> Kernel:
    return RBF(
        variance=rng.gamma(5.0, 0.2, []),
        lengthscales=rng.gamma(5.0, 0.2, [data.D]),
    )


def create_inducing(data: XYData, rng: np.random.Generator) -> InducingPointsLike:
    n = min(data.N // 2, 200)
    Z = Halton(data.D, scramble=False).random(n)
    lower = np.min(data.X, axis=0)
    upper = np.max(data.X, axis=0)
    Z = Z * (upper - lower) + lower
    return InducingPoints(Z)


def create_gaussian(data: XYData, rng: np.random.Generator) -> Gaussian:
    return Gaussian(variance=rng.gamma(5.0, 0.2, []))


@make_model_factory(tags={REGRESSION}, dataset_req=ds.REGRESSION & ~ds.LARGE)
def gpr(data: XYData, rng: np.random.Generator) -> GPModel:
    return GPR(
        data.XY,
        kernel=create_rbf(data, rng),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )


@make_model_factory(tags={REGRESSION, VARIATIONAL}, dataset_req=ds.REGRESSION & ~ds.LARGE)
def vgp(data: XYData, rng: np.random.Generator) -> GPModel:
    return VGP(
        data.XY,
        kernel=create_rbf(data, rng),
        likelihood=create_gaussian(data, rng),
    )


@make_model_factory(tags={REGRESSION, SPARSE}, dataset_req=ds.REGRESSION)
def sgpr(data: XYData, rng: np.random.Generator) -> GPModel:
    return SGPR(
        data.XY,
        kernel=create_rbf(data, rng),
        inducing_variable=create_inducing(data, rng),
        noise_variance=rng.gamma(5.0, 0.2, []),
    )


@make_model_factory(tags={REGRESSION, SPARSE, VARIATIONAL}, dataset_req=ds.REGRESSION)
def svgp(data: XYData, rng: np.random.Generator) -> GPModel:
    return SVGP(
        kernel=create_rbf(data, rng),
        likelihood=create_gaussian(data, rng),
        inducing_variable=create_inducing(data, rng),
    )
