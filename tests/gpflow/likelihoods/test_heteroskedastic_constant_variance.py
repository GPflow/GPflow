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

import gpflow
from gpflow.likelihoods import HeteroskedasticTFPConditional

tf.random.set_seed(99012)


class Data:
    g_var = 0.345
    rng = np.random.RandomState(123)
    N = 5
    Y = rng.randn(N, 1)
    # single "GP" (for the mean):
    f_mean = rng.randn(N, 1)
    f_var = rng.randn(N, 1) ** 2  # ensure positivity
    equivalent_f2 = np.log(g_var) / 2
    f2_mean = np.full((N, 1), equivalent_f2)
    f2_var = np.zeros((N, 1))
    F2_mean = np.c_[f_mean, f2_mean]
    F2_var = np.c_[f_var, f2_var]


def student_t_class_factory(df: int = 3) -> Type[tfp.distributions.StudentT]:
    r"""
    Returns tfp.Distribution.StudentT class (not instance!)
    where df (degrees of freedom) is pre-specified.

    This class allows to instantiate a StundentT object by passing
    loc and sale at initialisation for a given degree-of-freedom.
    """

    class _StudentT(tfp.distributions.StudentT):
        def __init__(self, loc, scale):
            super().__init__(df, loc=loc, scale=scale)

    return _StudentT


@pytest.fixture(name="equivalent_likelihoods", params=["studentt", "gaussian"])
def _equivant_likelihoods_fixture(
    request,
) -> Tuple[gpflow.likelihoods.ScalarLikelihood, gpflow.likelihoods.HeteroskedasticTFPConditional]:
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


def test_log_prob(equivalent_likelihoods):
    """
    heteroskedastic likelihood where the variance parameter is always constant
     giving the same answers for variational_expectations, predict_mean_and_var,
      etc as the regular Gaussian  likelihood
    """
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.log_prob(Data.f_mean, Data.Y),
        heteroskedastic_likelihood.log_prob(Data.F2_mean, Data.Y),
    )


def test_variational_expectations(equivalent_likelihoods):
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.variational_expectations(Data.f_mean, Data.f_var, Data.Y),
        heteroskedastic_likelihood.variational_expectations(Data.F2_mean, Data.F2_var, Data.Y),
        decimal=2,  # student-t case has a max absolute difference of 0.0034
    )


def test_predict_mean_and_var(equivalent_likelihoods):
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.predict_mean_and_var(Data.f_mean, Data.f_var),
        heteroskedastic_likelihood.predict_mean_and_var(Data.F2_mean, Data.F2_var),
    )


def test_conditional_mean(equivalent_likelihoods):
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.conditional_mean(Data.f_mean),
        heteroskedastic_likelihood.conditional_mean(Data.F2_mean),
    )


def test_conditional_variance(equivalent_likelihoods):
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    np.testing.assert_allclose(
        homoskedastic_likelihood.conditional_variance(Data.f_mean),
        heteroskedastic_likelihood.conditional_variance(Data.F2_mean),
    )


def test_predict_log_density(equivalent_likelihoods):
    homoskedastic_likelihood, heteroskedastic_likelihood = equivalent_likelihoods
    ll1 = homoskedastic_likelihood.predict_log_density(Data.f_mean, Data.f_var, Data.Y)
    ll2 = heteroskedastic_likelihood.predict_log_density(Data.F2_mean, Data.F2_var, Data.Y)
    np.testing.assert_array_almost_equal(
        homoskedastic_likelihood.predict_log_density(Data.f_mean, Data.f_var, Data.Y),
        heteroskedastic_likelihood.predict_log_density(Data.F2_mean, Data.F2_var, Data.Y),
        decimal=1,  # student-t has a max absolute difference of 0.025
    )
