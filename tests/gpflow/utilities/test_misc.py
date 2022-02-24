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
from typing import Any

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

from gpflow.base import Parameter
from gpflow.utilities import is_variable


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
