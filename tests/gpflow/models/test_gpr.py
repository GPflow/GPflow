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
import tensorflow as tf

import gpflow
from gpflow import set_trainable

rng = np.random.RandomState(0)


class Data:
    N = 10
    D = 1
    X = rng.rand(N, D)
    Y = rng.rand(N, 1)
    ls = 2.0
    var = 1.0


def test_non_trainable_model_objective() -> None:
    """
    Checks that we can still compute the objective of a model that has no
    trainable parameters whatsoever (regression test for bug in log_prior()).
    In this case we have no priors, so log_prior should be zero to add no
    contribution to the objective.
    """
    model = gpflow.models.GPR(
        (Data.X, Data.Y),
        kernel=gpflow.kernels.SquaredExponential(lengthscales=Data.ls, variance=Data.var),
    )

    set_trainable(model, False)

    _ = model.log_marginal_likelihood()
    assert model.log_prior_density() == 0.0


def test_varying_data() -> None:
    input_dim = 2
    output_dim = 1

    N = 5
    X, Y = rng.randn(N, input_dim), rng.randn(N, output_dim)

    var_data = (tf.Variable(X, shape=[None, input_dim]), tf.Variable(Y, shape=[None, output_dim]))
    m = gpflow.models.GPR(var_data, gpflow.kernels.SquaredExponential())

    lml_func = tf.function(m.log_marginal_likelihood)
    old_lml = lml_func()

    new_N = 7
    new_X, new_Y = rng.randn(new_N, input_dim), rng.randn(new_N, output_dim)

    # assign new data:
    for var, new_value in zip(var_data, (new_X, new_Y)):
        var.assign(new_value)

    new_lml = lml_func()  # re-use compiled function

    assert (
        np.abs((old_lml - new_lml) / (new_lml + old_lml)) > 0.1
    ), "we expect the LML for different data to be significantly different"
