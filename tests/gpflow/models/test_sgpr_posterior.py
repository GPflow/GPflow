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
""" Tests the SGPR posterior. """

from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.inducing_variables import InducingPoints
from gpflow.models.sgpr import SGPR, SGPR_deprecated
from gpflow.models.training_mixins import InputData, OutputData
from gpflow.posteriors import PrecomputeCacheType

INPUT_DIM = 7
OUTPUT_DIM = 1

MEAN_FUNCTION = gpflow.mean_functions.Constant(c=1.0)
KERNEL = gpflow.kernels.Matern52()
Z = np.random.randn(20, INPUT_DIM)


@pytest.fixture(name="dummy_data", scope="module")
def _dummy_data() -> Tuple[InputData, InputData, OutputData]:
    X = tf.convert_to_tensor(np.random.randn(100, INPUT_DIM), dtype=tf.float64)
    Y = tf.convert_to_tensor(np.random.randn(100, OUTPUT_DIM), dtype=tf.float64)
    X_new = tf.convert_to_tensor(np.random.randn(100, INPUT_DIM), dtype=tf.float64)

    return X, X_new, Y


@pytest.fixture(name="sgpr_deprecated_model", scope="module")
def _sgpr_deprecated_model(dummy_data) -> SGPR_deprecated:
    X, _, Y = dummy_data
    return SGPR_deprecated(
        data=(X, Y), kernel=KERNEL, inducing_variable=InducingPoints(Z), mean_function=MEAN_FUNCTION
    )


@pytest.fixture(name="sgpr_model", scope="module")
def sgpr_model(dummy_data) -> SGPR:
    X, _, Y = dummy_data
    return SGPR(
        data=(X, Y), kernel=KERNEL, inducing_variable=InducingPoints(Z), mean_function=MEAN_FUNCTION
    )


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_old_vs_new_gp_fused(
    sgpr_deprecated_model: SGPR_deprecated,
    sgpr_model: SGPR,
    dummy_data,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    _, X_new, _ = dummy_data

    mu_old, var2_old = sgpr_deprecated_model.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    mu_new_fuse, var2_new_fuse = sgpr_model.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new fuse is same as old version
    np.testing.assert_allclose(mu_new_fuse, mu_old)
    np.testing.assert_allclose(var2_new_fuse, var2_old)


# TODO: move to common test_model_utils
@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_old_vs_new_with_posterior(
    sgpr_deprecated_model: SGPR_deprecated,
    sgpr_model: SGPR,
    dummy_data,
    cache_type: PrecomputeCacheType,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    _, X_new, _ = dummy_data

    mu_old, var2_old = sgpr_deprecated_model.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    mu_new_cache, var2_new_cache = sgpr_model.posterior(cache_type).predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_old, mu_new_cache)
    np.testing.assert_allclose(var2_old, var2_new_cache)
