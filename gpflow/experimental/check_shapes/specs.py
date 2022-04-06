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
class ParsedNoteSpec:
    note: str

    def __repr__(self) -> str:
        return f"# {self.note}"


@dataclass(frozen=True)
class ParsedDimensionSpec:
    constant: Optional[int]
    variable_name: Optional[str]
    variable_rank: bool

    def __post_init__(self) -> None:
        assert (
            self.variable_name is None or self.constant is None
        ), "Dimension cannot be both constant and variable."
        if self.variable_rank:
            assert self.constant is None, "Constants cannot have a variable rank."

    def __repr__(self) -> str:
        if self.constant is not None:
            return str(self.constant)
        elif self.variable_rank:
            variable_name = self.variable_name or ""
            return variable_name + "..."
        else:
            return self.variable_name or "."


@dataclass(frozen=True)
class ParsedShapeSpec:
    dims: Tuple[ParsedDimensionSpec, ...]

    def __post_init__(self) -> None:
        n_variable_rank = sum(dim.variable_rank for dim in self.dims)
        assert (
            n_variable_rank <= 1
        ), f"At most one variable-rank dimension allowed. Found {n_variable_rank} in {self}."

    def __repr__(self) -> str:
        dims = [repr(dim) for dim in self.dims]
        return f"[{', '.join(dims)}]"


@dataclass(frozen=True)
class ParsedTensorSpec:
    shape: ParsedShapeSpec
    note: Optional[ParsedNoteSpec]

    def __repr__(self) -> str:
        note_str = f"  {self.note}" if self.note is not None else ""
        return f"{self.shape}{note_str}"


@dataclass(frozen=True)
class ParsedArgumentSpec:
    argument_ref: ArgumentRef
    tensor: ParsedTensorSpec

    def __repr__(self) -> str:
        return f"{self.argument_ref}: {self.tensor}"


@dataclass(frozen=True)
class ParsedFunctionSpec:
    arguments: Tuple[ParsedArgumentSpec, ...]
    notes: Tuple[ParsedNoteSpec, ...]

    def __repr__(self) -> str:
        lines = [repr(argument) for argument in self.arguments] + [
            repr(note) for note in self.notes
        ]
        return "\n".join(lines)
