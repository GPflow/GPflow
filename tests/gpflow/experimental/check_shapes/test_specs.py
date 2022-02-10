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

from gpflow.experimental.check_shapes.argument_ref import RootArgumentRef
from gpflow.experimental.check_shapes.specs import (
    ArgumentSpec,
    ParsedArgumentSpec,
    ParsedDimensionSpec,
    ParsedShapeSpec,
    parse_specs,
)


def match_dimension_spec(expected: ParsedDimensionSpec, actual: ParsedDimensionSpec) -> None:
    assert expected.constant == actual.constant
    assert expected.variable_name == actual.variable_name


def match_shape_spec(expected: ParsedShapeSpec, actual: ParsedShapeSpec) -> None:
    assert expected.leading_dims_variable_name == actual.leading_dims_variable_name
    assert len(expected.trailing_dims) == len(actual.trailing_dims)
    for e, a in zip(expected.trailing_dims, actual.trailing_dims):
        match_dimension_spec(e, a)


def match_argument_spec(expected: ParsedArgumentSpec, actual: ParsedArgumentSpec) -> None:
    assert repr(expected.argument_ref) == repr(actual.argument_ref)
    match_shape_spec(expected.shape, actual.shape)


@pytest.mark.parametrize(
    "raw_spec,expected,expected_repr",
    [
        (
            ("foo", []),
            ParsedArgumentSpec(RootArgumentRef("foo"), ParsedShapeSpec(None, ())),
            "foo: ()",
        ),
        (
            ("foo", [1, 2]),
            ParsedArgumentSpec(
                RootArgumentRef("foo"),
                ParsedShapeSpec(None, (ParsedDimensionSpec(1, None), ParsedDimensionSpec(2, None))),
            ),
            "foo: (1, 2)",
        ),
        (
            ("foo", ["x", "y"]),
            ParsedArgumentSpec(
                RootArgumentRef("foo"),
                ParsedShapeSpec(
                    None, (ParsedDimensionSpec(None, "x"), ParsedDimensionSpec(None, "y"))
                ),
            ),
            "foo: (x, y)",
        ),
        (
            ("foo", ["x...", "y"]),
            ParsedArgumentSpec(
                RootArgumentRef("foo"), ParsedShapeSpec("x", (ParsedDimensionSpec(None, "y"),))
            ),
            "foo: (x..., y)",
        ),
    ],
)
def test_parse_specs(
    raw_spec: ArgumentSpec, expected: ParsedArgumentSpec, expected_repr: str
) -> None:
    (parsed_spec,) = parse_specs([raw_spec])
    match_argument_spec(parsed_spec, expected)
    assert repr(parsed_spec) == expected_repr
