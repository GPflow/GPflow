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

import gpflow
from gpflow.config import default_float


def test_sparse_mcmc_likelihoods_and_gradients() -> None:
    """
    This test makes sure that when the inducing points are the same as the data
    points, the sparse mcmc is the same as full mcmc
    """
    rng = np.random.RandomState(0)
    X, Y = rng.randn(10, 1), rng.randn(10, 1)
    v_vals = rng.randn(10, 1)

    likelihood = gpflow.likelihoods.StudentT()
    model_1 = gpflow.models.GPMC(
        data=(X, Y), kernel=gpflow.kernels.Exponential(), likelihood=likelihood
    )
    model_2 = gpflow.models.SGPMC(
        data=(X, Y),
        kernel=gpflow.kernels.Exponential(),
        inducing_variable=X.copy(),
        likelihood=likelihood,
    )
    model_1.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_2.V = tf.convert_to_tensor(v_vals, dtype=default_float())
    model_1.kernel.lengthscales.assign(0.8)
    model_2.kernel.lengthscales.assign(0.8)
    model_1.kernel.variance.assign(4.2)
    model_2.kernel.variance.assign(4.2)

    assert_allclose(
        model_1.log_posterior_density(), model_2.log_posterior_density(), rtol=1e-5, atol=1e-5
    )
