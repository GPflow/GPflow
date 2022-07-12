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

from typing import Tuple, Type

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from _pytest.fixtures import SubRequest

import gpflow
from gpflow.base import AnyNDArray, TensorType
from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.likelihoods import HeteroskedasticTFPConditional

tf.random.set_seed(99012)


EquivalentLikelihoods = Tuple[
    gpflow.likelihoods.ScalarLikelihood, gpflow.likelihoods.HeteroskedasticTFPConditional
]


class Data:
    cs = ShapeChecker().check_shape

    g_var = 0.345
    rng = np.random.RandomState(123)
    N = 5
    X = cs(rng.randn(N, 2), "[N, D]")
    Y = cs(rng.randn(N, 1), "[N, P]")
    # single "GP" (for the mean):
    f_mean = cs(rng.randn(N, 1), "[N, Q]")
    f_var: AnyNDArray = cs(rng.randn(N, 1) ** 2, "[N, Q]")  # ensure positivity
    equivalent_f2 = cs(np.log(g_var) / 2, "[]")
    f2_mean = cs(np.full((N, 1), equivalent_f2), "[N, Q]")
    f2_var = cs(np.zeros((N, 1)), "[N, Q]")
    F2_mean = cs(np.c_[f_mean, f2_mean], "[N, Q2]")
    F2_var = cs(np.c_[f_var, f2_var], "[N, Q2]")


def student_t_class_factory(df: int = 3) -> Type[tfp.distributions.StudentT]:
    r"""
    Returns tfp.Distribution.StudentT class (not instance!)
    where df (degrees of freedom) is pre-specified.

    This class allows to instantiate a StundentT object by passing
    loc and sale at initialisation for a given degree-of-freedom.
    """

    class _StudentT(tfp.distributions.StudentT):
        def __init__(self, loc: TensorType, scale: TensorType) -> None:
            super().__init__(df, loc=loc, scale=scale)

    return _StudentT


@pytest.fixture(name="equivalent_likelihoods", params=["studentt", "gaussian"])
def _equivalent_likelihoods_fixture(
    request: SubRequest,
) -> EquivalentLikelihoods:
    if request.param == "studentt":
        return (
            gpflow.likelihoods.StudentT(scale=Data.g_var ** 0.5, df=3.0),
            HeteroskedasticTFPConditional(distribution_class=student_t_class_factory(df=3)),
        )
    elif request.param == "gaussian":
        return (
            gpflow.likelihoods.Gaussian(variance=Data.g_var),
            HeteroskedasticTFPConditional(distribution_class=tfp.distributions.Normal),
        )
    assert False, f"Unknown likelihood {request.param}."


def test_log_prob(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    """
    heteroskedastic likelihood where the variance parameter is always constant
     giving the same answers for variational_expectations, predict_mean_and_var,
      etc as the regular Gaussian  likelihood
    """
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.log_prob(Data.X, Data.f_mean, Data.Y),
        heteroskedastic_likelihood.log_prob(Data.X, Data.F2_mean, Data.Y),
    )


def test_variational_expectations(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.variational_expectations(Data.X, Data.f_mean, Data.f_var, Data.Y),
        heteroskedastic_likelihood.variational_expectations(
            Data.X, Data.F2_mean, Data.F2_var, Data.Y
        ),
        decimal=2,  # student-t case has a max absolute difference of 0.0034
    )


def test_predict_mean_and_var(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.predict_mean_and_var(Data.X, Data.f_mean, Data.f_var),
        heteroskedastic_likelihood.predict_mean_and_var(Data.X, Data.F2_mean, Data.F2_var),
    )


def test_conditional_mean(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.conditional_mean(Data.X, Data.f_mean),
        heteroskedastic_likelihood.conditional_mean(Data.X, Data.F2_mean),
    )


def test_conditional_variance(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.conditional_variance(Data.X, Data.f_mean),
        heteroskedastic_likelihood.conditional_variance(Data.X, Data.F2_mean),
    )


def test_predict_log_density(equivalent_likelihoods: EquivalentLikelihoods) -> None:
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.predict_log_density(Data.X, Data.f_mean, Data.f_var, Data.Y),
        heteroskedastic_likelihood.predict_log_density(Data.X, Data.F2_mean, Data.F2_var, Data.Y),
        decimal=1,  # student-t has a max absolute difference of 0.025
    )
