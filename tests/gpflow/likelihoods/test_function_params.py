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

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import tensorflow as tf

from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.functions import Linear
from gpflow.likelihoods import Beta, Gamma, Gaussian, ScalarLikelihood, StudentT
from gpflow.utilities import to_default_float


class Datum:
    cs = ShapeChecker().check_shapes
    rng = np.random.default_rng(20220623)

    batch = (1, 2)
    N = 5
    D = 3
    Q = 2
    X_shape = (*batch, N, D)
    Y_shape = (*batch, N, Q)

    np_X_positive = rng.random(X_shape)
    np_X_positive[:, :, :, 0] = np.linspace(0.1, 1.0, N)[None, None, :]
    X_positive = to_default_float(np_X_positive)
    X_negative = to_default_float(-rng.random(X_shape))
    F = to_default_float(0.5 * np.ones(Y_shape))
    Fmu = to_default_float(0.5 * np.ones(Y_shape))
    Fvar = to_default_float(0.1 * np.ones(Y_shape))
    Y = to_default_float(0.5 * np.ones(Y_shape))

    linear = Linear(A=[[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], b=0.0)


def diff(x: tf.Tensor, *, axis: int) -> tf.Tensor:
    s = [slice(None, None) for _ in x.shape]
    s[axis] = slice(None, -1)
    head = x.__getitem__(s)
    s[axis] = slice(1, None)
    tail = x.__getitem__(s)
    return tail - head


def assert_decreasing(x: tf.Tensor, *, axis: int) -> None:
    decreasing = diff(x, axis=axis) < 0
    assert tf.reduce_all(decreasing), x


def assert_increasing(x: tf.Tensor, *, axis: int) -> None:
    increasing = diff(x, axis=axis) > 0
    assert tf.reduce_all(increasing), x


def assert_constant(x: tf.Tensor, *, axis: int) -> None:
    same = diff(x, axis=axis) == 0
    assert tf.reduce_all(same), x


def no_assert(x: tf.Tensor, *, axis: int) -> None:
    pass


@dataclass
class LikelihoodSetup:
    name: str
    likelihood: ScalarLikelihood
    likelihood_assert: Any
    mean_assert: Any
    variance_assert: Any
    variational_expectations_assert: Any

    @property
    def __name__(self) -> str:  # Used by pytest to generate (pretty) test ids.
        return self.name


LIKELIHOODS = [
    LikelihoodSetup(
        name="gaussian_variance",
        likelihood=Gaussian(variance=Datum.linear),
        likelihood_assert=assert_decreasing,
        mean_assert=assert_constant,
        variance_assert=assert_increasing,
        variational_expectations_assert=assert_decreasing,
    ),
    LikelihoodSetup(
        name="gaussian_scale",
        likelihood=Gaussian(scale=Datum.linear),
        likelihood_assert=assert_decreasing,
        mean_assert=assert_constant,
        variance_assert=assert_increasing,
        variational_expectations_assert=no_assert,
    ),
    LikelihoodSetup(
        name="student_t",
        likelihood=StudentT(scale=Datum.linear),
        likelihood_assert=assert_decreasing,
        mean_assert=assert_constant,
        variance_assert=assert_increasing,
        variational_expectations_assert=no_assert,
    ),
    LikelihoodSetup(
        name="gamma",
        likelihood=Gamma(shape=Datum.linear),
        likelihood_assert=no_assert,
        mean_assert=assert_increasing,
        variance_assert=assert_increasing,
        variational_expectations_assert=no_assert,
    ),
    LikelihoodSetup(
        name="beta",
        likelihood=Beta(scale=Datum.linear),
        likelihood_assert=no_assert,
        mean_assert=assert_constant,
        variance_assert=assert_decreasing,
        variational_expectations_assert=assert_increasing,
    ),
]


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_log_prob__positive(setup: LikelihoodSetup) -> None:
    lp = setup.likelihood.log_prob(Datum.X_positive, Datum.F, Datum.Y)
    setup.likelihood_assert(lp, axis=-1)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_log_prob__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    lp = setup.likelihood.log_prob(Datum.X_negative, Datum.F, Datum.Y)
    assert_constant(lp, axis=-1)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_conditional_mean__positive(setup: LikelihoodSetup) -> None:
    # We don't expect X to affect the mean.
    cm = setup.likelihood.conditional_mean(Datum.X_positive, Datum.F)
    setup.mean_assert(cm, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_conditional_mean__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    cm = setup.likelihood.conditional_mean(Datum.X_negative, Datum.F)
    assert_constant(cm, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_conditional_variance__positive(setup: LikelihoodSetup) -> None:
    cv = setup.likelihood.conditional_variance(Datum.X_positive, Datum.F)
    setup.variance_assert(cv, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_conditional_variance__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    cv = setup.likelihood.conditional_variance(Datum.X_negative, Datum.F)
    assert_constant(cv, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_predict_mean_and_var__positive(setup: LikelihoodSetup) -> None:
    mu, var = setup.likelihood.predict_mean_and_var(Datum.X_positive, Datum.Fmu, Datum.Fvar)
    setup.mean_assert(mu, axis=-2)
    setup.variance_assert(var, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_predict_mean_and_var__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    mu, var = setup.likelihood.predict_mean_and_var(Datum.X_negative, Datum.Fmu, Datum.Fvar)
    assert_constant(mu, axis=-2)
    assert_constant(var, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_predict_log_density__positive(setup: LikelihoodSetup) -> None:
    ld = setup.likelihood.predict_log_density(Datum.X_positive, Datum.Fmu, Datum.Fvar, Datum.Y)
    setup.likelihood_assert(ld, axis=-1)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_predict_log_density__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    ld = setup.likelihood.predict_log_density(Datum.X_negative, Datum.Fmu, Datum.Fvar, Datum.Y)
    assert_constant(ld, axis=-2)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_variational_expectation__positive(setup: LikelihoodSetup) -> None:
    ve = setup.likelihood.variational_expectations(Datum.X_positive, Datum.Fmu, Datum.Fvar, Datum.Y)
    setup.variational_expectations_assert(ve, axis=-1)


@pytest.mark.parametrize("setup", LIKELIHOODS)
def test_variational_expectation__negative(setup: LikelihoodSetup) -> None:
    # We expect the negative parameter values to be clamped to the same value, so output should be
    # constant:
    ve = setup.likelihood.variational_expectations(Datum.X_negative, Datum.Fmu, Datum.Fvar, Datum.Y)
    assert_constant(ve, axis=-2)
