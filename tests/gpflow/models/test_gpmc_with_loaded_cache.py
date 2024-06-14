# Copyright 2016-2020 the GPflow authors.
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

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
from check_shapes import check_shapes
from numpy.testing import assert_allclose
from tensorflow_probability import distributions as tfd

import gpflow
from gpflow.base import AnyNDArray
from gpflow.config import default_float
from gpflow.models.gpmc import GPMC_deprecated, GPMC_with_posterior
from gpflow.posteriors import PrecomputeCacheType

f64 = gpflow.utilities.to_default_float


@check_shapes(
    "return[0]: [N, D]  # X",
    "return[1]: [batch_new..., N_new, D]  # X_new",
    "return[2]: [N, P]  # Y",
    "return[3]: [N, 1]  # v_vals",
)
def _get_data_for_tests() -> Tuple[AnyNDArray, AnyNDArray, AnyNDArray, AnyNDArray]:
    """Helper function to create testing data"""
    X = np.random.randn(10, 1)
    Y = np.random.randn(10, 1)
    v_vals = np.random.randn(10, 1)

    X_new = np.random.randn(3, 10, 10, 1)
    return X, X_new, Y, v_vals


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_predict_f_vs_predict_f_loaded_cache(full_cov: bool, full_output_cov: bool) -> None:
    """
    This test makes sure that when the data points and the values of hyper-parameters are the same,
    the faster prediction with loaded cashe will return the same results as the standard prediction method.
    """
    X, X_new, Y, v_vals = _get_data_for_tests()

    likelihood = gpflow.likelihoods.StudentT()
    model_1 = GPMC_deprecated(
        data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood
    )
    model_2 = GPMC_with_posterior(
        data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood
    )

    model_1.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_2.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_1.kernel.lengthscales.assign(0.8)
    model_2.kernel.lengthscales.assign(0.8)
    model_1.kernel.variance.assign(4.2)
    model_2.kernel.variance.assign(4.2)
    stored_cache = model_2.posterior().cache

    mu_f_old, var2_f_old = model_1.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    mu_f_cache, var2_f_cache = model_2.predict_f_loaded_cache(
        X_new, Cache=stored_cache, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_f_old, mu_f_cache)
    np.testing.assert_allclose(var2_f_old, var2_f_cache)


@pytest.mark.parametrize("full_cov", [False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_predict_y_vs_predict_y_loaded_cache(full_cov: bool, full_output_cov: bool) -> None:
    """
    This test makes sure that when the data points and the values of hyper-parameters are the same,
    the faster prediction with loaded cashe will return the same results as the standard prediction method.
    """
    X, X_new, Y, v_vals = _get_data_for_tests()

    likelihood = gpflow.likelihoods.StudentT()
    model_1 = GPMC_deprecated(
        data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood
    )
    model_2 = GPMC_with_posterior(
        data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood
    )

    model_1.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_2.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_1.kernel.lengthscales.assign(0.8)
    model_2.kernel.lengthscales.assign(0.8)
    model_1.kernel.variance.assign(4.2)
    model_2.kernel.variance.assign(4.2)
    stored_cache = model_2.posterior().cache
    mu_y_old, var2_y_old = model_1.predict_y(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    mu_y_cache, var2_y_cache = model_2.predict_y_loaded_cache(
        X_new, Cache=stored_cache, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_y_old, mu_y_cache)
    np.testing.assert_allclose(var2_y_old, var2_y_cache)
