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

import numpy as np
import tensorflow as tf

from gpflow.base import AnyNDArray
from gpflow.likelihoods import HeteroskedasticTFPConditional

tf.random.set_seed(99012)


class Data:
    rng = np.random.RandomState(123)
    N = 5
    X = np.linspace(0, 1, num=N)[:, None]
    Y = rng.randn(N, 1)
    f_mean = rng.randn(N, 2)
    f_var: AnyNDArray = rng.randn(N, 2) ** 2


def test_analytic_mean_and_var() -> None:
    """
    Test that quadrature computation used in HeteroskedasticTFPConditional
    of the predictive mean and variance is close to the analytical version,
    which can be computed for the special case of N(y | mean=f1, scale=exp(f2)),
    where f1, f2 ~ GP.
    """
    analytic_mean = Data.f_mean[:, [0]]
    analytic_variance = np.exp(Data.f_mean[:, [1]] + Data.f_var[:, [1]]) ** 2 + Data.f_var[:, [0]]

    likelihood = HeteroskedasticTFPConditional()
    y_mean, y_var = likelihood.predict_mean_and_var(Data.X, Data.f_mean, Data.f_var)

    np.testing.assert_allclose(y_mean, analytic_mean)
    np.testing.assert_allclose(y_var, analytic_variance, rtol=1.5e-6)
