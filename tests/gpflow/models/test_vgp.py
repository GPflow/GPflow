# Copyright 2022 the GPflow authors.
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
from gpflow.config import default_float


def test_update_vgp_data() -> None:
    rng = np.random.default_rng(20220223)
    sample = lambda *shape: tf.convert_to_tensor(rng.standard_normal(shape), dtype=default_float())

    n_inputs = 2
    n_outputs = 1
    n_data_1 = 3
    n_data_2 = 2

    X_1 = tf.Variable(sample(n_data_1, n_inputs), shape=(None, n_inputs), trainable=False)
    Y_1 = tf.Variable(sample(n_data_1, n_outputs), shape=(None, n_outputs), trainable=False)

    model = gpflow.models.VGP(
        (X_1, Y_1),
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        num_latent_gps=n_outputs,
    )
    gpflow.optimizers.Scipy().minimize(
        model.training_loss_closure(),
        variables=model.trainable_variables,
        options=dict(maxiter=25),
        compile=True,
    )

    X_test = tf.constant(tf.convert_to_tensor(X_1))
    mean_before, var_before = model.predict_f(X_test)

    X_2 = sample(n_data_2, n_inputs)
    Y_2 = sample(n_data_2, n_outputs)
    gpflow.models.vgp.update_vgp_data(
        model, (tf.concat([X_1, X_2], axis=0), tf.concat([Y_1, Y_2], axis=0))
    )

    (
        mean_after,
        var_after,
    ) = model.predict_f(X_test)

    np.testing.assert_allclose(mean_before, mean_after, atol=1e-5)
    np.testing.assert_allclose(var_before, var_after, atol=1e-6)
