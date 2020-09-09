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
    rng = np.random.RandomState(123)
    N = 5
    Y = rng.randn(N, 1)
    # single "GP" (for the mean):
    f_mean = rng.randn(N, 2)
    f_var = rng.randn(N, 2) ** 2


# @pytest.fixture
# def likelihood() -> HeteroskedasticTFPConditional:
#     return HeteroskedasticTFPConditional(num_gauss_hermite_points=200)


def test_analytic_mean_and_var():
    """
    Test that quadrature computation used in HeteroskedasticTFPConditional
    of the predictive mean and variance is close to the analytical version, 
    which can be computed for the special case of N(y | f1, exp(f2)),
    where f1, f2 ~ GP.
    """
    likelihood = HeteroskedasticTFPConditional(num_gauss_hermite_points=200)
    analytic_mean = Data.f_mean[:, [0]]
    analytic_variance = np.exp(Data.f_mean[:, [1]] + Data.f_var[:, [1]] / 2) + Data.f_var[:, [0]]

    y_mean, y_var = likelihood.predict_mean_and_var(Data.f_mean, Data.f_var)
    np.testing.assert_allclose(y_mean, analytic_mean)
    np.testing.assert_allclose(y_var, analytic_variance)

