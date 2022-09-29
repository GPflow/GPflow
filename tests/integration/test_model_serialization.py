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

from pathlib import Path

import numpy as np
import tensorflow as tf

import gpflow


def test_model_serialization(tmp_path: Path) -> None:
    rng = np.random.default_rng(1234)
    X = rng.random((5, 1))
    Y = rng.standard_normal(X.shape)
    kernel = gpflow.kernels.SquaredExponential()
    model = gpflow.models.GPR((X, Y), kernel=kernel)

    frozen_model = gpflow.utilities.freeze(model)
    module_out = tf.Module()
    module_out.predict = tf.function(
        frozen_model.predict_y,
        input_signature=[tf.TensorSpec(shape=[None, 1], dtype=tf.float64)],
    )

    tf.saved_model.save(module_out, str(tmp_path))

    module_in = tf.saved_model.load(str(tmp_path))

    Xnew = rng.random((3, 1))
    np.testing.assert_allclose(
        model.predict_y(Xnew),
        module_in.predict(Xnew),
    )
