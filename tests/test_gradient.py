# Copyright 2018 the GPflow authors.
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
from gpflow.test_util import session_tf

def test_columnwise_gradient(session_tf):
    X = tf.placeholder(gpflow.settings.float_type, [None, None])
    X_np = np.random.randn(6, 2)
    Y = tf.reduce_sum(X ** 2, axis=1)
    Y = tf.reshape(Y, [-1, 1])
    grads = gpflow.misc.columnwise_gradients(Y, X)
    grads = tf.reshape(grads, [6, 2])
    expected = 2 * X_np
    res = session_tf.run(grads, feed_dict={X: X_np})
    assert_allclose(res, expected, atol=1e-10)

