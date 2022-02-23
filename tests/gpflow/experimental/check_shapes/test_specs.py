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

from gpflow.experimental.check_shapes.specs import ParsedArgumentSpec

from .utils import make_argument_ref, make_shape_spec, varrank


@pytest.mark.parametrize(
    "argument_spec,expected_repr",
    [
        (
            ParsedArgumentSpec(
                make_argument_ref("foo"),
                make_shape_spec(),
            ),
            "foo: ()",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("foo"),
                make_shape_spec(1, 2),
            ),
            "foo: (1, 2)",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("foo"),
                make_shape_spec("x", "y"),
            ),
            "foo: (x, y)",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("foo"),
                make_shape_spec(varrank("x"), "y"),
            ),
            "foo: (x..., y)",
        ),
        (
            ParsedArgumentSpec(
                make_argument_ref("foo"),
                make_shape_spec("x", varrank("y"), "z"),
            ),
            "foo: (x, y..., z)",
        ),
    ],
)
def test_specs(argument_spec: ParsedArgumentSpec, expected_repr: str) -> None:
    assert expected_repr == repr(argument_spec)
