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
"""
Utilities for testing the `check_shapes` library.
"""
from typing import Optional, Union
from unittest.mock import MagicMock

from gpflow.base import TensorType
from gpflow.experimental.check_shapes.argument_ref import (
    ArgumentRef,
    AttributeArgumentRef,
    IndexArgumentRef,
    RootArgumentRef,
)
from gpflow.experimental.check_shapes.specs import ParsedDimensionSpec, ParsedShapeSpec


def t(*shape: Optional[int]) -> TensorType:
    """
    Creates a mock tensor of the given shape.
    """
    mock_tensor = MagicMock()
    mock_tensor.shape = shape
    return mock_tensor


def make_shape_spec(
    leading_dims_variable_name: Optional[str], *dims: Union[int, str]
) -> ParsedShapeSpec:
    shape = []
    for dim in dims:
        if isinstance(dim, int):
            shape.append(ParsedDimensionSpec(constant=dim, variable_name=None))
        else:
            shape.append(ParsedDimensionSpec(constant=None, variable_name=dim))
    return ParsedShapeSpec(leading_dims_variable_name, tuple(shape))


def make_argument_ref(argument_name: str, *refs: Union[int, str]) -> ArgumentRef:
    result: ArgumentRef = RootArgumentRef(argument_name)
    for ref in refs:
        if isinstance(ref, int):
            result = IndexArgumentRef(result, ref)
        else:
            result = AttributeArgumentRef(result, ref)
    return result
