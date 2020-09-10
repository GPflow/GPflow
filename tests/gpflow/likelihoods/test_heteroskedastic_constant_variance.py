# Copyright 2017-2020 the GPflow authors.
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
from typing import Type, Tuple

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
    equivalent_f2 = np.log(g_var)
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


# @pytest.mark.skip("Conditional mean is not implemented in heteroskedastic likelihood")
# def test_conditional_mean():
#     l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
#     l2 = HeteroskedasticTFPConditional()
#     np.testing.assert_allclose(
#         l1.conditional_mean(Data.f_mean), l2.conditional_mean(Data.F2_mean),
#     )


# @pytest.mark.skip("Conditional variance is not implemented in heteroskedastic likelihood")
# def test_conditional_variance():
#     l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
#     l2 = HeteroskedasticTFPConditional()
#     np.testing.assert_allclose(
#         l1.conditional_variance(Data.f_mean), l2.conditional_variance(Data.F2_mean),
#     )


# @pytest.mark.skip("Currently broken as it returns the sum over outputs when given multiple outputs")
# def test_predict_log_density():
#     l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
#     l2 = HeteroskedasticTFPConditional()
#     np.testing.assert_allclose(
#         l1.predict_log_density(Data.f_mean, Data.f_var, Data.Y),
#         l2.predict_log_density(Data.F2_mean, Data.f2_var, Data.Y),
#     )
