# Copyright 2022 The GPflow Contributors. All Rights Reserved.
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
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import Parameter
from gpflow.experimental.check_shapes import Shape, get_shape
from gpflow.experimental.check_shapes.exceptions import NoShapeError

from .utils import TestContext


def make_tensor_coercible(
    shape: Shape, concrete: bool
) -> tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible:
    loc = tf.zeros(shape)
    scale = tf.ones(shape)
    dist = tfp.python.layers.internal.distribution_tensor_coercible._TensorCoercible(
        tfp.distributions.Normal(loc, scale), lambda self: loc
    )
    if concrete:
        tf.convert_to_tensor(dist)  # Triggers some caching within `dist`.
    return dist


@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (True, ()),
        (0, ()),
        (0.0, ()),
        ("foo", ()),
        ((), None),
        ((0,), (1,)),
        ([[0.1, 0.2]], (1, 2)),
        ([[[], []]], None),
        (np.zeros(()), ()),
        (np.zeros((3, 4)), (3, 4)),
        (tf.zeros(()), ()),
        (tf.zeros((4, 3)), (4, 3)),
        (tf.Variable(np.zeros(())), ()),
        (tf.Variable(np.zeros((2, 4))), (2, 4)),
        # pylint: disable=unexpected-keyword-arg
        (tf.Variable(np.zeros((2, 4)), shape=[2, None]), (2, None)),
        (tf.Variable(np.zeros((2, 4)), shape=tf.TensorShape(None)), None),
        (Parameter(3), ()),
        (Parameter(np.zeros((4, 2))), (4, 2)),
        (make_tensor_coercible((), True), ()),
        (make_tensor_coercible((4, 5), True), (4, 5)),
        (make_tensor_coercible((), False), None),
        (make_tensor_coercible((4, 5), False), None),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Shape) -> None:
    actual_shape = get_shape(shaped, TestContext())

    # Numpy and tensorflow sometimes like to pretend that objects have another type than they
    # actually do. Make sure the result actually has the right types:
    if actual_shape is not None:
        assert tuple == type(actual_shape)
        assert all((actual_dim is None) or (int == type(actual_dim)) for actual_dim in actual_shape)

    assert expected_shape == actual_shape


@pytest.mark.parametrize(
    "shaped,expected_message",
    [
        (
            object(),
            """
Unable to determine shape of object.
  Fake test error context.
    Object type: builtins.object
""",
        ),
        (
            [[object()]],
            """
Unable to determine shape of object.
  Fake test error context.
    Index: [0]
      Index: [0]
        Object type: builtins.object
""",
        ),
    ],
)
def test_get_shape__error(shaped: Any, expected_message: str) -> None:
    with pytest.raises(NoShapeError) as e:
        get_shape(shaped, TestContext())

    (message,) = e.value.args
    assert expected_message == message
