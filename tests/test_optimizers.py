# Copyright 2019 the GPflow authors.
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
from numpy.testing import assert_allclose

import gpflow
from gpflow.config import default_jitter
from gpflow.mean_functions import Constant

rng = np.random.RandomState(0)


class Datum:
    X = rng.rand(20, 1) * 10
    Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
    Y = np.tile(Y, 2)  # two identical columns
    Xtest = rng.rand(10, 1) * 10
    data = (X, Y)


def _create_full_gp_model():
    """
    GP Regression
    """
    return gpflow.models.GPR(
        (Datum.X, Datum.Y),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=gpflow.mean_functions.Constant(),
    )

def test_scipy_jit():
    m1 = _create_full_gp_model()
    m2 = _create_full_gp_model()

    opt1 = gpflow.optimizers.Scipy()
    opt2 = gpflow.optimizers.Scipy()

    def closure1():
        return - m1.log_marginal_likelihood()

    @tf.function(autograph=False)
    def closure2():
        return - m2.log_marginal_likelihood()

    opt1.minimize(closure1, variables=m1.trainable_variables, options=dict(maxiter=50), jit=False)
    opt2.minimize(closure2, variables=m2.trainable_variables, options=dict(maxiter=50), jit=True)

    def get_values(model):
        return np.array([var.value().numpy().squeeze() for var in model.trainable_variables])

    np.testing.assert_allclose(get_values(m1), get_values(m2), rtol=1e-14, atol=1e-15)
