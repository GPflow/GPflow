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
import pytest

from gpflow.experimental.check_shapes.specs import (
    ParsedArgumentSpec,
    ParsedFunctionSpec,
    ParsedNoteSpec,
)

from .utils import bc, make_arg_spec, make_argument_ref, make_shape_spec, varrank


def test_note_spec() -> None:
    note_spec = ParsedNoteSpec("foo")
    assert "# foo" == repr(note_spec)


@pytest.mark.parametrize(
    "argument_spec,expected_repr",
    [
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(),
                note=None,
            ),
            "foo: []",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(1, 2),
                note=None,
            ),
            "foo: [1, 2]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec("x", "y"),
                note=None,
            ),
            "foo: [x, y]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(varrank("x"), "y"),
                note=None,
            ),
            "foo: [x..., y]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec("x", varrank("y"), "z"),
                note=None,
            ),
            "foo: [x, y..., z]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(None, varrank(None), None),
                note=None,
            ),
            "foo: [., ..., .]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(bc(varrank("y")), bc(3), bc("x")),
                note=None,
            ),
            "foo: [broadcast y..., broadcast 3, broadcast x]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec(bc(varrank(None)), bc(None)),
                note=None,
            ),
            "foo: [broadcast ..., broadcast .]",
        ),
        (
            make_arg_spec(
                make_argument_ref("foo"),
                make_shape_spec("x", "y"),
                note=ParsedNoteSpec("bar"),
            ),
            "foo: [x, y]  # bar",
        ),
    ],
)
def test_argument_spec(argument_spec: ParsedArgumentSpec, expected_repr: str) -> None:
    assert expected_repr == repr(argument_spec)


@pytest.mark.parametrize(
    "function_spec,expected_repr",
    [
        (ParsedFunctionSpec((), ()), ""),
        (
            ParsedFunctionSpec(
                (
                    make_arg_spec(
                        make_argument_ref("foo"),
                        make_shape_spec("x", "y"),
                        note=None,
                    ),
                ),
                (ParsedNoteSpec("note 1"),),
            ),
            "foo: [x, y]\n# note 1",
        ),
        (
            ParsedFunctionSpec(
                (
                    make_arg_spec(
                        make_argument_ref("foo"),
                        make_shape_spec("x", "y"),
                        note=None,
                    ),
                    make_arg_spec(
                        make_argument_ref("bar"),
                        make_shape_spec("x", "z"),
                        note=ParsedNoteSpec("bar note"),
                    ),
                ),
                (
                    ParsedNoteSpec("note 1"),
                    ParsedNoteSpec("note 2"),
                ),
            ),
            """foo: [x, y]
bar: [x, z]  # bar note
# note 1
# note 2""",
        ),
    ],
)
def test_function_spec(function_spec: ParsedFunctionSpec, expected_repr: str) -> None:
    assert expected_repr == repr(function_spec)
