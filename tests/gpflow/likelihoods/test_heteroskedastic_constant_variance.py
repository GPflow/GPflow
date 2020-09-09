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
    equivalent_f2 = np.log(np.sqrt(g_var))
    f2_mean = np.full((N, 1), equivalent_f2)
    f2_var = np.zeros((N, 1))
    F2_mean = np.c_[f_mean, f2_mean]
    F2_var = np.c_[f_var, f2_var]


def test_log_prob():
    """
    heteroskedastic likelihood where the variance parameter is always constant
     giving the same answers for variational_expectations, predict_mean_and_var,
      etc as the regular Gaussian  likelihood
    """
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.log_prob(Data.f_mean, Data.Y), l2.log_prob(Data.F2_mean, Data.Y),
    )


def test_variational_expectations():
    # Create likelihoods
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.variational_expectations(Data.f_mean, Data.f_var, Data.Y),
        l2.variational_expectations(Data.F2_mean, Data.F2_var, Data.Y),
    )


def test_predict_mean_and_var():
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.predict_mean_and_var(Data.f_mean, Data.f_var),
        l2.predict_mean_and_var(Data.F2_mean, Data.F2_var),
    )


@pytest.mark.skip("Conditional mean is not implemented in heteroskedastic likelihood")
def test_conditional_mean():
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.conditional_mean(Data.f_mean), l2.conditional_mean(Data.F2_mean),
    )


@pytest.mark.skip("Conditional variance is not implemented in heteroskedastic likelihood")
def test_conditional_variance():
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.conditional_variance(Data.f_mean), l2.conditional_variance(Data.F2_mean),
    )


@pytest.mark.skip("Currently broken as it returns the sum over outputs when given multiple outputs")
def test_predict_log_density():
    l1 = gpflow.likelihoods.Gaussian(variance=Data.g_var)
    l2 = HeteroskedasticTFPConditional(tfp.distributions.Normal)
    np.testing.assert_allclose(
        l1.predict_log_density(Data.f_mean, Data.f_var, Data.Y),
        l2.predict_log_density(Data.F2_mean, Data.f2_var, Data.Y),
    )
