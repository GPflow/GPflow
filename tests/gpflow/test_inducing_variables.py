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

import gpflow


def test_inducing_points_with_variable_shape():
    N, M1, D, P = 50, 13, 3, 1
    X, Y = np.random.randn(N, D), np.random.randn(N, P)

    Z1 = np.random.randn(M1, D)

    iv = gpflow.inducing_variables.InducingPoints(Z1)
    # overwrite Parameter with explicit tf.Variable with None shape:
    iv.Z = tf.Variable(Z1, trainable=False, dtype=gpflow.default_float(), shape=(None, D))

    m = gpflow.models.SGPR(data=(X, Y), kernel=gpflow.kernels.Matern32(), inducing_variable=iv)

    opt = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        opt.minimize(m.training_loss, m.trainable_variables)

    optimization_step()

    # Check 1: that we can successfully assign a new Z with different number of inducing points!
    Z2 = np.random.randn(M1 + 1, D)
    m.inducing_variable.Z.assign(Z2)

    # Check 2: that we can still optimize!
    optimization_step()
