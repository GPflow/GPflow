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
from typing import Optional, Union

import tensorflow as tf

from ..base import Parameter, TensorData, TensorType
from ..experimental.check_shapes import check_shapes
from ..functions import Function
from . import positive

ConstantOrFunction = Union[Function, TensorData]
ParameterOrFunction = Union[Function, Parameter]


def prepare_parameter_or_function(
    value: ConstantOrFunction,
    *,
    lower_bound: Optional[float] = None,
) -> ParameterOrFunction:
    if isinstance(value, Function):
        return value
    else:
        if lower_bound is None:
            return Parameter(value)
        else:
            return Parameter(value, transform=positive(lower_bound))


@check_shapes(
    "X: [batch..., N, D]",
    "return: [broadcast batch..., broadcast N, broadcast P]",
)
def evaluate_parameter_or_function(
    value: ParameterOrFunction,
    X: TensorType,
    *,
    lower_bound: Optional[float] = None,
) -> TensorType:
    if isinstance(value, Function):
        result = value(X)
        if lower_bound is not None:
            result = tf.maximum(result, lower_bound)
        return result
    else:
        return value
