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
from typing import Any, Type, Union

import numpy as np
import pytest
import tensorflow as tf

from gpflow.base import Parameter
from gpflow.experimental.check_shapes import ActualShape, get_shape


@pytest.mark.parametrize(
    "shaped,expected_shape",
    [
        (object(), NotImplementedError),
        (True, ()),
        (0, ()),
        (0.0, ()),
        ("foo", NotImplementedError),
        ((), None),
        ((0,), (1,)),
        ([[0.1, 0.2]], (1, 2)),
        ([[[], []]], None),
        (np.zeros((3, 4)), (3, 4)),
        (tf.zeros((4, 3)), (4, 3)),
        (tf.Variable(np.zeros((2, 4))), (2, 4)),
        (tf.Variable(np.zeros((2, 4)), shape=[2, None]), (2, None)),
        (tf.Variable(np.zeros((2, 4)), shape=tf.TensorShape(None)), None),
        (Parameter(3), ()),
        (Parameter(np.zeros((4, 2))), (4, 2)),
    ],
)
def test_get_shape(shaped: Any, expected_shape: Union[ActualShape, Type[Exception]]) -> None:
    if isinstance(expected_shape, type) and issubclass(expected_shape, Exception):

        with pytest.raises(expected_shape):
            get_shape(shaped)

    else:

        actual_shape = get_shape(shaped)

        # Numpy and tensorflow sometimes like to pretend that objects have another type than they
        # actually do. Make sure the result actually has the right types:
        if actual_shape is not None:
            assert tuple == type(actual_shape)
            assert all(
                (actual_dim is None) or (int == type(actual_dim)) for actual_dim in actual_shape
            )

        assert expected_shape == actual_shape
