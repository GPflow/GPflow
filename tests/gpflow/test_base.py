# Copyright 2020 the GPflow authors.
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


import tempfile

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.utilities import positive


def test_parameter_assign_validation():
    with pytest.raises(tf.errors.InvalidArgumentError):
        param = gpflow.Parameter(0.0, transform=positive())

    param = gpflow.Parameter(0.1, transform=positive())
    param.assign(0.2)
    with pytest.raises(tf.errors.InvalidArgumentError):
        param.assign(0.0)


def test_cast_to_dtype_precision_issue():
    """
    TensorFlow's tf.cast(value, dtype) implicitly does a tf.convert_to_tensor(value)
    *before* the cast when the value is not a tensor already. When value is a python float,
    this results in the following behaviour:

    >>> tf.cast(0.2, tf.float64)
    <tf.Tensor: id=37, shape=(), dtype=float64, numpy=0.20000000298023224>
    
    instead of the expected expansion of 0.2 to float64 precision that you get when
    passing in an object that already carries dtype information, such as a numpy array
    (which has float64 precision by default):

    >>> tf.cast(np.array(0.2), tf.float64)
    <tf.Tensor: id=40, shape=(), dtype=float64, numpy=0.2>

    This affected *all* gpflow.Parameter objects, resulting in numerical discrepancies
    between GPflow 1 and 2, due to the pass through _cast_to_dtype, which is now fixed.
    This is the corresponding regression test.
    """
    p = gpflow.Parameter(0.2, dtype=np.float64)
    actual_value = p.numpy()
    assert actual_value.dtype == np.float64
    expected_value = np.float64(0.2)
    assert actual_value == expected_value


def test_parameter_saved():
    dtype = tf.float64

    class Model(tf.Module):
        def __init__(self):
            self.p = gpflow.Parameter(0.1, dtype=dtype, transform=gpflow.utilities.positive())

        @tf.function(input_signature=[tf.TensorSpec([], dtype=dtype)])
        def exec(self, x: tf.Tensor) -> tf.Tensor:
            return tf.square(x * self.p)

    m0 = Model()
    x = tf.convert_to_tensor(2.0, dtype=dtype)
    expected = m0.exec(x)
    with tempfile.TemporaryDirectory() as dirname:
        tf.saved_model.save(m0, dirname)
        m1 = tf.saved_model.load(dirname)
        actual = m1.exec(x)
        np.testing.assert_equal(actual, expected)
