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
from unittest.mock import MagicMock

from gpflow.experimental.check_shapes.argument_ref import RESULT_TOKEN
from gpflow.experimental.check_shapes.errors import ArgumentReferenceError, ShapeMismatchError
from gpflow.experimental.check_shapes.specs import parse_specs

from .utils import t


def context_func() -> None:
    pass


def test_argument_reference_error() -> None:
    arg_map = MagicMock()
    arg_ref = MagicMock(__str__=MagicMock(return_value="my_argument_reference"))

    error = ArgumentReferenceError(context_func, arg_map, arg_ref)

    assert error.func == context_func
    assert error.arg_map == arg_map
    assert error.arg_ref == arg_ref
    assert (
        str(error)
        == f"""Unable to resolve argument / missing argument.
    Function: context_func
    Declared: {__file__}:23
    Argument: my_argument_reference"""
    )


def test_shape_mismatch_error() -> None:
    specs = parse_specs(
        [
            ("a", ["x", 3]),
            ("b", [5, "y"]),
            (RESULT_TOKEN, ["x", "y"]),
        ]
    )
    arg_map = {
        "a": t(2, 3),
        "b": t(5, 6),
        RESULT_TOKEN: t(2, 6),
    }
    error = ShapeMismatchError(context_func, specs, arg_map)

    assert error.func == context_func
    assert error.specs == specs
    assert error.arg_map == arg_map
    assert (
        str(error)
        == f"""Tensor shape mismatch in call to function.
    Function: context_func
    Declared: {__file__}:23
    Argument: a, expected: (x, 3), actual: (2, 3)
    Argument: b, expected: (5, y), actual: (5, 6)
    Argument: return, expected: (x, y), actual: (2, 6)"""
    )
