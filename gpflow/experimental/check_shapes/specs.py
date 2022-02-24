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
Code for specifying expectations around shapes.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

from .argument_ref import ArgumentRef


@dataclass(frozen=True)
class ParsedDimensionSpec:
    constant: Optional[int]
    variable_name: Optional[str]

    def __post_init__(self) -> None:
        assert (self.constant is None) != (
            self.variable_name is None
        ), "Argument must be either constant or variable."

    def __repr__(self) -> str:
        if self.constant is not None:
            return str(self.constant)
        else:
            assert self.variable_name is not None
            return self.variable_name


@dataclass(frozen=True)
class ParsedShapeSpec:
    leading_dims_variable_name: Optional[str]
    trailing_dims: Tuple[ParsedDimensionSpec, ...]

    def __repr__(self) -> str:
        dims = []
        if self.leading_dims_variable_name:
            dims.append(f"{self.leading_dims_variable_name}...")
        dims.extend(repr(dim) for dim in self.trailing_dims)
        return f"({', '.join(dims)})"


@dataclass(frozen=True)
class ParsedArgumentSpec:
    argument_ref: ArgumentRef
    shape: ParsedShapeSpec

    def __repr__(self) -> str:
        return f"{self.argument_ref}: {self.shape}"
