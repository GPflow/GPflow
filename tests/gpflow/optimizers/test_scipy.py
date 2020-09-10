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

    opt1.minimize(
        m1.training_loss, variables=m1.trainable_variables, options=dict(maxiter=50), compile=False,
    )
    opt2.minimize(
        m2.training_loss, variables=m2.trainable_variables, options=dict(maxiter=50), compile=True,
    )

    def get_values(model):
        return np.array([var.numpy().squeeze() for var in model.trainable_variables])

    # The tolerance of the following test had to be loosened slightly from atol=1e-15
    # due to the changes introduced by PR #1213, which removed some implicit casts
    # to float32.
    np.testing.assert_allclose(get_values(m1), get_values(m2), rtol=1e-14, atol=1e-14)
