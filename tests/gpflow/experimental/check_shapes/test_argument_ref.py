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
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import pytest

from gpflow.experimental.check_shapes import ArgumentReferenceError
from gpflow.experimental.check_shapes.argument_ref import RESULT_TOKEN, parse_argument_ref


def context_func() -> None:
    pass


@dataclass
class SomeClass:
    a: str
    b: str


ARG_MAP: Mapping[str, Any] = {
    "foo": "foo_value",
    "bar": "bar_value",
    "some_class": SomeClass("aaa", "bbb"),
    "some_tuple": (42, 69),
    RESULT_TOKEN: "return_value",
}


@pytest.mark.parametrize(
    "argument_ref_str,expected_is_result,expected_get",
    [
        ("foo", False, "foo_value"),
        (RESULT_TOKEN, True, "return_value"),
        ("baz", None, ArgumentReferenceError),
        ("123", None, AssertionError),
        ("some_class.b", False, "bbb"),
        ("some_class.c", None, ArgumentReferenceError),
        ("some_class.1", None, AssertionError),
        ("some_tuple[1]", False, 69),
        ("some_tuple[2]", None, ArgumentReferenceError),
        ("some_tuple[a]", None, AssertionError),
        (".a", None, AssertionError),
        ("[0]", None, AssertionError),
        ("some_class..a", None, AssertionError),
        ("some_class.[0]", None, AssertionError),
    ],
)
def test_parse_argument_ref(
    argument_ref_str: str, expected_is_result: Optional[bool], expected_get: Any
) -> None:
    if isinstance(expected_get, type) and issubclass(expected_get, Exception):
        if issubclass(expected_get, ArgumentReferenceError):
            ref = parse_argument_ref(argument_ref_str)
            with pytest.raises(ArgumentReferenceError) as ex_info:
                ref.get(context_func, ARG_MAP)

            ex = ex_info.value
            assert ex.func == context_func
            assert ex.arg_map == ARG_MAP
            assert ex.arg_ref == ref
        else:
            with pytest.raises(expected_get):
                parse_argument_ref(argument_ref_str)
    else:
        ref = parse_argument_ref(argument_ref_str)
        assert ref.is_result == expected_is_result
        assert ref.get(context_func, ARG_MAP) == expected_get
        assert repr(ref) == argument_ref_str
