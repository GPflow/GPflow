# Copyright 2017-2021 The GPflow Contributors. All Rights Reserved.
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
from typing import Any, Union

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import Parameter
from gpflow.utilities import is_variable, to_default_float


@pytest.mark.parametrize(
    "value,expected",
    [
        ([[0.1, 0.2], [0.3, 0.4]], False),
        (np.array(5), False),
        (tf.constant(5), False),
        (tf.Variable(5), True),
        (tfp.util.TransformedVariable(5, tfp.bijectors.Identity()), True),
        (Parameter(5), True),
    ],
    ids=lambda x: x if isinstance(x, bool) else type(x).__name__,
)
def test_is_variable(value: Any, expected: bool) -> None:
    assert expected == is_variable(value)


@pytest.mark.parametrize(
    "x",
    [
        0.999999999,
    ],
)
def test_to_default_float(x: Union[float, tf.Tensor]) -> None:
    default_float_x = to_default_float(x)
    _x = x if isinstance(x, float) else float(x.numpy())
    assert default_float_x.numpy() == _x


@pytest.mark.parametrize(
    ("x", "expect_to_fail"),
    [
        (0.9999999, False),
        # This testcase will start to fail if tensorflow changes the behaviour
        # of `tf.cast( , dtype=tf.float64)` to convert python floats directly to float64 tensors.
        # However, there are good reasons for why it doesn't, so it probably won't happen.
        #
        # For more context, see this ticket:
        # https://github.com/tensorflow/tensorflow/issues/57779
        #
        # If tensorflow does make this change, then change `expect_to_fail`
        # to False, and remove the workaround involving `tf.convert_to_tensor`
        # from `to_default_float` because it is now obsolete.
        (0.99999999, True),
    ],
)
def test_tf_cast_precision(x: Union[float, tf.Tensor], expect_to_fail: bool) -> None:
    def assert_doesnt_round_to_one() -> None:
        float_64_x = tf.cast(x, dtype=tf.float64)
        assert float_64_x.numpy() != tf.convert_to_tensor(1.0, dtype=tf.float64)

    if expect_to_fail:
        with pytest.raises(AssertionError):
            assert_doesnt_round_to_one()
    else:
        assert_doesnt_round_to_one()
