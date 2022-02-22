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
from gpflow.experimental.check_shapes.argument_ref import RESULT_TOKEN, ArgumentRef

from .utils import make_argument_ref


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
    "argument_ref,expected_is_result,expected_root_argument_name,expected_repr,expected_get",
    [
        (make_argument_ref("foo"), False, "foo", "foo", "foo_value"),
        (make_argument_ref(RESULT_TOKEN), True, RESULT_TOKEN, RESULT_TOKEN, "return_value"),
        (make_argument_ref("baz"), False, "baz", "baz", ArgumentReferenceError),
        (make_argument_ref("some_class", "b"), False, "some_class", "some_class.b", "bbb"),
        (
            make_argument_ref("some_class", "c"),
            False,
            "some_class",
            "some_class.c",
            ArgumentReferenceError,
        ),
        (make_argument_ref("some_tuple", 1), False, "some_tuple", "some_tuple[1]", 69),
        (
            make_argument_ref("some_tuple", 2),
            False,
            "some_tuple",
            "some_tuple[2]",
            ArgumentReferenceError,
        ),
        (
            make_argument_ref(RESULT_TOKEN, 1, "a"),
            True,
            "return",
            "return[1].a",
            ArgumentReferenceError,
        ),
    ],
)
def test_parse_argument_ref(
    argument_ref: ArgumentRef,
    expected_is_result: Optional[bool],
    expected_root_argument_name: str,
    expected_repr: str,
    expected_get: Any,
) -> None:
    assert expected_is_result == argument_ref.is_result
    assert expected_root_argument_name == argument_ref.root_argument_name
    assert expected_repr == repr(argument_ref)

    if isinstance(expected_get, type) and issubclass(expected_get, Exception):
        assert issubclass(expected_get, ArgumentReferenceError)
        with pytest.raises(ArgumentReferenceError) as ex_info:
            argument_ref.get(context_func, ARG_MAP)

        ex = ex_info.value
        assert ex.func == context_func  # pylint: disable=comparison-with-callable
        assert ex.arg_map == ARG_MAP
        assert ex.arg_ref == argument_ref
    else:
        assert argument_ref.get(context_func, ARG_MAP) == expected_get
