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

# pylint: disable=broad-except

"""
Errors raised by `check_shapes`.
"""
import inspect
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import tensorflow as tf

from .base_types import C

if TYPE_CHECKING:
    from .argument_ref import ArgumentRef
    from .specs import ParsedArgumentSpec


class ArgumentReferenceError(Exception):
    """ Error raised if the argument to check the shape of could not be resolved. """

    def __init__(self, func: C, arg_map: Mapping[str, Any], arg_ref: "ArgumentRef") -> None:
        func_info = _FunctionDebugInfo.create(func)
        lines = [
            "Unable to resolve argument / missing argument.",
            f"    Function: {func_info.name}",
            f"    Declared: {func_info.path_and_line}",
            f"    Argument: {arg_ref}",
        ]

        super().__init__("\n".join(lines))

        self.func = func
        self.arg_map = arg_map
        self.arg_ref = arg_ref


class ShapeMismatchError(Exception):
    """ Error raised if a function is called with tensors of the wrong shape. """

    def __init__(
        self,
        func: C,
        specs: Sequence["ParsedArgumentSpec"],
        arg_map: Mapping[str, Any],
    ) -> None:
        func_info = _FunctionDebugInfo.create(func)
        lines = [
            "Tensor shape mismatch in call to function.",
            f"    Function: {func_info.name}",
            f"    Declared: {func_info.path_and_line}",
        ]
        for spec in specs:
            actual_shape = spec.argument_ref.get(func, arg_map).shape
            if isinstance(actual_shape, tf.TensorShape) and actual_shape.rank is None:
                actual_str = "<Unknown>"
            else:
                actual_str = f"({', '.join(str(dim) for dim in actual_shape)})"
            lines.append(
                f"    Argument: {spec.argument_ref}, expected: {spec.shape}, actual: {actual_str}"
            )

        super().__init__("\n".join(lines))

        self.func = func
        self.specs = specs
        self.arg_map = arg_map


@dataclass
class _FunctionDebugInfo:
    """
    Information about a function, to print in error messages.
    """

    name: str
    path_and_line: str

    @staticmethod
    def create(func: C) -> "_FunctionDebugInfo":
        name = func.__qualname__
        try:
            path = inspect.getsourcefile(func)
        except Exception:  # pragma: no cover
            path = "<unknown file>"
        try:
            _, line_int = inspect.getsourcelines(func)
            line = str(line_int)
        except Exception:  # pragma: no cover
            line = "<unknown lines>"
        path_and_line = f"{path}:{line}"

        return _FunctionDebugInfo(name=name, path_and_line=path_and_line)
