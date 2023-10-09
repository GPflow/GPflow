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

import numpy as np
import tensorflow as tf
from numpy.testing import assert_allclose
from check_shapes import check_shapes
from typing import Tuple
import pytest

import gpflow
from gpflow.config import default_float
from gpflow.models.gpr import GPR_deprecated, GPR_with_posterior
from gpflow.posteriors import PrecomputeCacheType
from gpflow.base import AnyNDArray


from tensorflow_probability import distributions as tfd
f64 = gpflow.utilities.to_default_float

@check_shapes(
    "return[0]: [N, D]  # X",
    "return[1]: [batch_new..., N_new, D]  # X_new",
    "return[2]: [N, P]  # Y",
)
def _get_data_for_tests() -> Tuple[AnyNDArray, AnyNDArray, AnyNDArray]:
    """Helper function to create testing data"""
    X = np.random.randn(10, 1)
    Y = np.random.randn(10, 1)
    X_new = np.random.randn(3, 10, 10, 1)
    return X, X_new, Y


@check_shapes(
    "regression_data[0]: [N, D]",
    "regression_data[1]: [N, P]",
)
def make_models(
    regression_data: gpflow.base.RegressionData,
) -> Tuple[GPR_deprecated, GPR_with_posterior]:
    """Helper function to create models"""

    mold = GPR_deprecated(data=regression_data, kernel=gpflow.kernels.Exponential())
    mold.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mold.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mold.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mold.kernel.lengthscales.assign(0.8)
    mold.kernel.variance.assign(4.2)

    mnew = GPR_with_posterior(data=regression_data, kernel=gpflow.kernels.Exponential())
    mnew.kernel.lengthscales.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mnew.kernel.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mnew.likelihood.variance.prior = tfd.Gamma(f64(1.0), f64(1.0))
    mnew.kernel.lengthscales.assign(0.8)
    mnew.kernel.variance.assign(4.2)
    
    return mold, mnew



@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_predict_f_vs_predict_f_loaded_cache(
        full_cov: bool, full_output_cov: bool
) -> None:
    """
    This test makes sure that when the data points and the values of hyper-parameters are the same,
    the faster prediction with loaded cashe will return the same results as the standard prediction method.
    """
    X, X_new, Y = _get_data_for_tests()
    model_1, model_2 = make_models((X, Y))

    stored_cache = model_2.posterior().cache

    mu_f_old, var2_f_old = model_1.predict_f(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_f_cache, var2_f_cache = model_2.predict_f_loaded_cache(X_new, Cache=stored_cache, full_cov=full_cov, full_output_cov=full_output_cov)

    # check new cache is same as old version
    np.testing.assert_allclose(mu_f_old, mu_f_cache)
    np.testing.assert_allclose(var2_f_old, var2_f_cache)



@pytest.mark.parametrize("full_cov", [False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_predict_y_vs_predict_y_loaded_cache(
        full_cov: bool, full_output_cov: bool
) -> None:
    """
    This test makes sure that when the data points and the values of hyper-parameters are the same,
    the faster prediction with loaded cashe will return the same results as the standard prediction method.
    """
    X, X_new, Y = _get_data_for_tests()

    model_1, model_2 = make_models((X, Y))
    stored_cache = model_2.posterior().cache

    mu_y_old, var2_y_old = model_1.predict_y(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_y_cache, var2_y_cache = model_2.predict_y_loaded_cache(X_new, Cache=stored_cache, full_cov=full_cov, full_output_cov=full_output_cov)

    # check new cache is same as old version
    np.testing.assert_allclose(mu_y_old, mu_y_cache)
    np.testing.assert_allclose(var2_y_old, var2_y_cache)