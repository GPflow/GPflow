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
from gpflow.config import default_float, default_jitter
from gpflow.mean_functions import Constant

rng = np.random.RandomState(0)


class Datum:
    X = rng.rand(20, 1) * 10
    Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
    Y = np.tile(Y, 2)  # two identical columns
    Xtest = rng.rand(10, 1) * 10
    data = (X, Y)

    # for classification:
    Yc = Y[:, :1]
    cdata = (X, Yc)


@pytest.mark.parametrize("whiten", [True, False])
@pytest.mark.parametrize("q_diag", [True, False])
def test_svgp(whiten, q_diag):
    model = gpflow.models.SVGP(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        inducing_variable=Datum.X.copy(),
        q_diag=q_diag,
        whiten=whiten,
        mean_function=gpflow.mean_functions.Constant(),
        num_latent_gps=Datum.Y.shape[1],
    )
    gpflow.set_trainable(model.inducing_variable, False)

    # test with explicitly unknown shapes:
    tensor_spec = tf.TensorSpec(shape=None, dtype=default_float())
    elbo = tf.function(model.elbo, input_signature=[(tensor_spec, tensor_spec)],)

    @tf.function
    def model_closure():
        return -elbo(Datum.data)

    opt = gpflow.optimizers.Scipy()

    # simply test whether it runs without erroring...:
    opt.minimize(
        model_closure, variables=model.trainable_variables, options=dict(maxiter=3), compile=True,
    )


def test_multiclass():
    num_classes = 3
    model = gpflow.models.SVGP(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.MultiClass(num_classes=num_classes),
        inducing_variable=Datum.X.copy(),
        num_latent_gps=num_classes,
    )
    gpflow.set_trainable(model.inducing_variable, False)

    # test with explicitly unknown shapes:
    tensor_spec = tf.TensorSpec(shape=None, dtype=default_float())
    elbo = tf.function(model.elbo, input_signature=[(tensor_spec, tensor_spec)],)

    @tf.function
    def model_closure():
        return -elbo(Datum.cdata)

    opt = gpflow.optimizers.Scipy()

    # simply test whether it runs without erroring...:
    opt.minimize(
        model_closure, variables=model.trainable_variables, options=dict(maxiter=3), compile=True,
    )
