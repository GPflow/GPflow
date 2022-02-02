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
import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from .argument_ref import ArgumentRef, parse_argument_ref

DimensionSpec = Union[int, str]
ShapeSpec = Sequence[DimensionSpec]
ArgumentSpec = Tuple[str, ShapeSpec]

_NAME_RE_STR = "([_a-zA-Z][_a-zA-Z0-9]*)"
_ELLIPSIS_RE_STR = r"(\.\.\.)"
_VARIABLE_RE = re.compile(f"{_NAME_RE_STR}{_ELLIPSIS_RE_STR}?")


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


def parse_spec(raw_spec: ArgumentSpec) -> ParsedArgumentSpec:
    argument_ref_str, raw_shape_spec = raw_spec

    argument_ref = parse_argument_ref(argument_ref_str)

    leading_dims_variable_name: Optional[str] = None
    trailing_dims: List[ParsedDimensionSpec] = []
    is_first = True
    for i, raw_dimension_spec in enumerate(raw_shape_spec):
        if isinstance(raw_dimension_spec, int):
            trailing_dims.append(
                ParsedDimensionSpec(constant=raw_dimension_spec, variable_name=None)
            )
        else:
            assert isinstance(
                raw_dimension_spec, str
            ), f"Invalid dimension specification type {type(raw_dimension_spec)}."
            match = _VARIABLE_RE.fullmatch(raw_dimension_spec)
            assert match, f"Invalid dimension specification {raw_dimension_spec}."
            if match[2] is None:
                trailing_dims.append(ParsedDimensionSpec(constant=None, variable_name=match[1]))
            else:
                assert is_first, (
                    "Only the leading dimension can have variable length."
                    f" Found variable length for argument {argument_ref}, dimension {i}."
                )
                leading_dims_variable_name = match[1]
        is_first = False
    shape_spec = ParsedShapeSpec(leading_dims_variable_name, tuple(trailing_dims))
    return ParsedArgumentSpec(argument_ref, shape_spec)


def parse_specs(raw_specs: Sequence[ArgumentSpec]) -> Sequence[ParsedArgumentSpec]:
    return [parse_spec(raw_spec) for raw_spec in raw_specs]
