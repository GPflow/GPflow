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

from typing import Sequence, Tuple

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
import gpflow.ci_utils
from gpflow.base import TensorType
from gpflow.config import default_float, default_int

from gpflow.likelihoods import (  # isort:skip
    # classes we cannot test:
    HeteroskedasticTFPConditional,
    Likelihood,
    MonteCarloLikelihood,
    MultiLatentLikelihood,
    MultiLatentTFPConditional,
    QuadratureLikelihood,
    ScalarLikelihood,
    # classes we do test in this file:
    Bernoulli,
    Beta,
    Exponential,
    Gamma,
    Gaussian,
    GaussianMC,
    MultiClass,
    Ordinal,
    Poisson,
    StudentT,
)

tf.random.set_seed(99012)


class Datum:
    tolerance = 1e-06
    N = 10
    Xshape = (N, 2)
    Yshape = (N, 3)
    X = tf.random.normal(Xshape, dtype=tf.float64)
    Y = tf.random.normal(Yshape, dtype=tf.float64)
    F = tf.random.normal(Yshape, dtype=tf.float64)
    Fmu = tf.random.normal(Yshape, dtype=tf.float64)
    Fvar = 0.01 * tf.random.normal(Yshape, dtype=tf.float64) ** 2
    Fvar_zero = tf.zeros(Yshape, dtype=tf.float64)


class LikelihoodSetup(object):
    def __init__(
        self,
        likelihood: Likelihood,
        Y: TensorType = Datum.Y,
        rtol: float = 1e-06,
        atol: float = 0.0,
    ) -> None:
        self.likelihood = likelihood
        self.Y = Y
        self.rtol = rtol
        self.atol = atol

    def __repr__(self) -> str:
        name = self.likelihood.__class__.__name__
        return f"{name}-rtol={self.rtol}-atol={self.atol}"


scalar_likelihood_setups = [
    LikelihoodSetup(Gaussian()),
    LikelihoodSetup(StudentT()),
    LikelihoodSetup(Beta(), Y=tf.random.uniform(Datum.Yshape, dtype=default_float())),
    LikelihoodSetup(
        Ordinal(np.array([-1, 1])),
        Y=tf.random.uniform(Datum.Yshape, 0, 3, dtype=default_int()),
    ),
    LikelihoodSetup(
        Poisson(invlink=tf.square),
        Y=tf.random.poisson(Datum.Yshape, 1.0, dtype=default_float()),
    ),
    LikelihoodSetup(
        Exponential(invlink=tf.square),
        Y=tf.random.uniform(Datum.Yshape, dtype=default_float()),
    ),
    LikelihoodSetup(
        Gamma(invlink=tf.square),
        Y=tf.random.uniform(Datum.Yshape, dtype=default_float()),
    ),
    LikelihoodSetup(
        Bernoulli(invlink=tf.sigmoid),
        Y=tf.random.uniform(Datum.Yshape, dtype=default_float()),
    ),
]

likelihood_setups = scalar_likelihood_setups + [
    LikelihoodSetup(
        MultiClass(3),
        Y=tf.argmax(Datum.Y, 1).numpy().reshape(-1, 1),
        rtol=1e-3,
        atol=1e-3,
    ),
]


def filter_analytic_scalar_likelihood(method_name: str) -> Sequence[LikelihoodSetup]:
    assert method_name in (
        "_variational_expectations",
        "_predict_log_density",
        "_predict_mean_and_var",
    )

    def is_analytic(likelihood: Likelihood) -> bool:
        assert not isinstance(likelihood, MonteCarloLikelihood)
        assert isinstance(likelihood, ScalarLikelihood)
        quadrature_fallback = getattr(ScalarLikelihood, method_name)
        actual_method = getattr(likelihood.__class__, method_name)
        return actual_method is not quadrature_fallback

    return [l for l in scalar_likelihood_setups if is_analytic(get_likelihood(l))]


def get_likelihood(likelihood_setup: LikelihoodSetup) -> Likelihood:
    if isinstance(likelihood_setup, type(pytest.param())):
        (likelihood_setup,) = likelihood_setup.values
    return likelihood_setup.likelihood


def test_no_missing_likelihoods() -> None:
    tested_likelihood_types = [get_likelihood(l).__class__ for l in likelihood_setups]
    for likelihood_class in gpflow.ci_utils.subclasses(Likelihood):
        if likelihood_class in (
            QuadratureLikelihood,
            ScalarLikelihood,
            MonteCarloLikelihood,
            MultiLatentLikelihood,
        ):
            # abstract base classes that cannot be tested
            continue

        if likelihood_class in tested_likelihood_types:
            # tested by parametrized tests (see test_multiclass.py for MultiClass quadrature)
            continue

        if likelihood_class is gpflow.likelihoods.SwitchedLikelihood:
            # tested separately, see test_switched_likelihood.py
            continue

        if likelihood_class in (MultiLatentTFPConditional, HeteroskedasticTFPConditional):
            # tested separately, see test_heteroskedastic*.py
            continue

        if issubclass(likelihood_class, MonteCarloLikelihood):
            if likelihood_class is GaussianMC:
                continue  # tested explicitly by test_montecarlo_*
            if likelihood_class is gpflow.likelihoods.Softmax:
                continue  # tested explicitly by test_softmax_{y_shape_assert,bernoulli_equivalence}, see test_multiclass.py

        assert False, f"no test for likelihood class {likelihood_class}"


@pytest.mark.parametrize("likelihood_setup", likelihood_setups)
@pytest.mark.parametrize("X, mu, var", [[Datum.X, Datum.Fmu, tf.zeros_like(Datum.Fmu)]])
def test_conditional_mean_and_variance(
    likelihood_setup: LikelihoodSetup, X: TensorType, mu: TensorType, var: TensorType
) -> None:
    """
    Here we make sure that the conditional_mean and conditional_var functions
    give the same result as the predict_mean_and_var function if the prediction
    has no uncertainty.
    """
    mu1 = likelihood_setup.likelihood.conditional_mean(X, mu)
    var1 = likelihood_setup.likelihood.conditional_variance(X, mu)
    mu2, var2 = likelihood_setup.likelihood.predict_mean_and_var(X, mu, var)
    assert_allclose(mu1, mu2, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)
    assert_allclose(var1, var2, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)


@pytest.mark.parametrize("likelihood_setup", likelihood_setups)
def test_variational_expectations(likelihood_setup: LikelihoodSetup) -> None:
    """
    Here we make sure that the variational_expectations gives the same result
    as log_prob if the latent function has no uncertainty.
    """
    likelihood = likelihood_setup.likelihood
    X = Datum.X
    F = Datum.F
    Y = likelihood_setup.Y
    r1 = likelihood.log_prob(X, F, Y)
    r2 = likelihood.variational_expectations(X, F, tf.zeros_like(F), Y)
    assert_allclose(r1, r2, atol=likelihood_setup.atol, rtol=likelihood_setup.rtol)


@pytest.mark.parametrize(
    "likelihood_setup", filter_analytic_scalar_likelihood("_variational_expectations")
)
@pytest.mark.parametrize("mu, var", [[Datum.Fmu, Datum.Fvar]])
def test_scalar_likelihood_quadrature_variational_expectation(
    likelihood_setup: LikelihoodSetup, mu: TensorType, var: TensorType
) -> None:
    """
    Where quadrature methods have been overwritten, make sure the new code
    does something close to the quadrature.
    """
    x = Datum.X
    likelihood, y = likelihood_setup.likelihood, likelihood_setup.Y
    F1 = likelihood.variational_expectations(x, mu, var, y)
    F2 = ScalarLikelihood.variational_expectations(likelihood, x, mu, var, y)
    assert_allclose(F1, F2, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)


@pytest.mark.parametrize(
    "likelihood_setup", filter_analytic_scalar_likelihood("_predict_log_density")
)
@pytest.mark.parametrize("mu, var", [[Datum.Fmu, Datum.Fvar]])
def test_scalar_likelihood_quadrature_predict_log_density(
    likelihood_setup: LikelihoodSetup, mu: TensorType, var: TensorType
) -> None:
    x = Datum.X
    likelihood, y = likelihood_setup.likelihood, likelihood_setup.Y
    F1 = likelihood.predict_log_density(x, mu, var, y)
    F2 = ScalarLikelihood.predict_log_density(likelihood, x, mu, var, y)
    assert_allclose(F1, F2, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)


@pytest.mark.parametrize(
    "likelihood_setup", filter_analytic_scalar_likelihood("_predict_mean_and_var")
)
@pytest.mark.parametrize("X, mu, var", [[Datum.X, Datum.Fmu, Datum.Fvar]])
def test_scalar_likelihood_quadrature_predict_mean_and_var(
    likelihood_setup: LikelihoodSetup, X: TensorType, mu: TensorType, var: TensorType
) -> None:
    likelihood = likelihood_setup.likelihood
    F1m, F1v = likelihood.predict_mean_and_var(X, mu, var)
    F2m, F2v = ScalarLikelihood.predict_mean_and_var(likelihood, X, mu, var)
    assert_allclose(F1m, F2m, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)
    assert_allclose(F1v, F2v, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)


def _make_montecarlo_x_mu_var_y() -> Sequence[tf.Tensor]:
    x_mu_var_y = [tf.random.normal((3, 10), dtype=tf.float64)] * 4
    x_mu_var_y[2] = 0.01 * (x_mu_var_y[2] ** 2)
    return x_mu_var_y


def _make_montecarlo_likelihoods(var: float) -> Tuple[GaussianMC, Gaussian]:
    gaussian_mc_likelihood = GaussianMC(var)
    gaussian_mc_likelihood.num_monte_carlo_points = 1000000
    return gaussian_mc_likelihood, Gaussian(var)


@pytest.mark.parametrize("likelihood_var", [0.3, 0.5, 1])
@pytest.mark.parametrize("x, mu, var, y", [_make_montecarlo_x_mu_var_y()])
def test_montecarlo_variational_expectation(
    likelihood_var: float, x: TensorType, mu: TensorType, var: TensorType, y: TensorType
) -> None:
    likelihood_gaussian_mc, likelihood_gaussian = _make_montecarlo_likelihoods(likelihood_var)
    assert_allclose(
        likelihood_gaussian_mc.variational_expectations(x, mu, var, y),
        likelihood_gaussian.variational_expectations(x, mu, var, y),
        rtol=5e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("likelihood_var", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("x, mu, var, y", [_make_montecarlo_x_mu_var_y()])
def test_montecarlo_predict_log_density(
    likelihood_var: float, x: TensorType, mu: TensorType, var: TensorType, y: TensorType
) -> None:
    likelihood_gaussian_mc, likelihood_gaussian = _make_montecarlo_likelihoods(likelihood_var)
    assert_allclose(
        likelihood_gaussian_mc.predict_log_density(x, mu, var, y),
        likelihood_gaussian.predict_log_density(x, mu, var, y),
        rtol=5e-4,
        atol=1e-4,
    )


@pytest.mark.parametrize("likelihood_var", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("x, mu, var, y", [_make_montecarlo_x_mu_var_y()])
def test_montecarlo_predict_mean_and_var(
    likelihood_var: float, x: TensorType, mu: TensorType, var: TensorType, y: TensorType
) -> None:
    likelihood_gaussian_mc, likelihood_gaussian = _make_montecarlo_likelihoods(likelihood_var)
    mean1, var1 = likelihood_gaussian_mc.predict_mean_and_var(x, mu, var)
    mean2, var2 = likelihood_gaussian.predict_mean_and_var(x, mu, var)
    assert_allclose(mean1, mean2, rtol=5e-4, atol=1e-4)
    assert_allclose(var1, var2, rtol=5e-4, atol=1e-4)
