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
from typing import Any, Mapping, Optional, Sequence, Tuple

import pytest

from gpflow.experimental.check_shapes.argument_ref import RESULT_TOKEN, ArgumentRef
from gpflow.experimental.check_shapes.error_contexts import (
    ArgumentContext,
    AttributeContext,
    ErrorContext,
    IndexContext,
    MappingKeyContext,
    MappingValueContext,
    StackContext,
)
from gpflow.experimental.check_shapes.exceptions import ArgumentReferenceError

from .utils import TestContext, all_ref, keys_ref, make_argument_ref, values_ref


@dataclass(frozen=True)
class SomeClass:
    a: Optional[str]
    b: str


@dataclass(frozen=True)
class ManyClass:
    many: Sequence[Optional[int]]


ARG_MAP: Mapping[str, Any] = {
    "foo": "foo_value",
    "bar": "bar_value",
    "none": None,
    "some_class": SomeClass("aaa", "bbb"),
    "some_class_none": SomeClass(None, "bbb"),
    "some_tuple": (42, 69),
    "some_tuple_none": (None, None),
    "many": [
        (
            3,
            {
                "a": ManyClass(many=[111, 112]),
                "b": ManyClass(many=[121, 122]),
            },
        ),
        None,
        (
            7,
            {
                None: ManyClass(many=[None, 221]),
                "d": None,
            },
        ),
    ],
    RESULT_TOKEN: "return_value",
}


@dataclass(frozen=True)
class ArgumentRefTest:
    name: str
    argument_ref: ArgumentRef
    expected_is_result: Optional[bool]
    expected_root_argument_name: str
    expected_repr: str
    expected_get: Sequence[Tuple[Any, ErrorContext]] = ()
    expected_error: Optional[str] = None


def name(test: ArgumentRefTest) -> str:
    return test.name


sc = StackContext


TESTS = [
    ArgumentRefTest(
        name="argument",
        argument_ref=make_argument_ref("foo"),
        expected_is_result=False,
        expected_root_argument_name="foo",
        expected_repr="foo",
        expected_get=[("foo_value", ArgumentContext("foo"))],
    ),
    ArgumentRefTest(
        name="return",
        argument_ref=make_argument_ref(RESULT_TOKEN),
        expected_is_result=True,
        expected_root_argument_name=RESULT_TOKEN,
        expected_repr=RESULT_TOKEN,
        expected_get=[("return_value", ArgumentContext(RESULT_TOKEN))],
    ),
    ArgumentRefTest(
        name="None_argument",
        argument_ref=make_argument_ref("none"),
        expected_is_result=False,
        expected_root_argument_name="none",
        expected_repr="none",
        expected_get=[(None, ArgumentContext("none"))],
    ),
    ArgumentRefTest(
        name="missing_argument",
        argument_ref=make_argument_ref("baz"),
        expected_is_result=False,
        expected_root_argument_name="baz",
        expected_repr="baz",
        expected_error="""
Unable to resolve argument / missing argument.
  Fake test error context.
    Argument: baz
""",
    ),
    ArgumentRefTest(
        name="class_member",
        argument_ref=make_argument_ref("some_class", "b"),
        expected_is_result=False,
        expected_root_argument_name="some_class",
        expected_repr="some_class.b",
        expected_get=[("bbb", sc(ArgumentContext("some_class"), AttributeContext("b")))],
    ),
    ArgumentRefTest(
        name="None_class_member",
        argument_ref=make_argument_ref("none", "c"),
        expected_is_result=False,
        expected_root_argument_name="none",
        expected_repr="none.c",
        expected_get=[(None, ArgumentContext("none"))],
    ),
    ArgumentRefTest(
        name="class_None_member",
        argument_ref=make_argument_ref("some_class_none", "a"),
        expected_is_result=False,
        expected_root_argument_name="some_class_none",
        expected_repr="some_class_none.a",
        expected_get=[(None, sc(ArgumentContext("some_class_none"), AttributeContext("a")))],
    ),
    ArgumentRefTest(
        name="class_missing_member",
        argument_ref=make_argument_ref("some_class", "c"),
        expected_is_result=False,
        expected_root_argument_name="some_class",
        expected_repr="some_class.c",
        expected_error="""
Unable to resolve argument / missing argument.
  Fake test error context.
    Argument: some_class
      Attribute: .c
""",
    ),
    ArgumentRefTest(
        name="tuple_element",
        argument_ref=make_argument_ref("some_tuple", 1),
        expected_is_result=False,
        expected_root_argument_name="some_tuple",
        expected_repr="some_tuple[1]",
        expected_get=[(69, sc(ArgumentContext("some_tuple"), IndexContext(1)))],
    ),
    ArgumentRefTest(
        name="None_tuple_element",
        argument_ref=make_argument_ref("none", 1),
        expected_is_result=False,
        expected_root_argument_name="none",
        expected_repr="none[1]",
        expected_get=[(None, ArgumentContext("none"))],
    ),
    ArgumentRefTest(
        name="tuple_None_element",
        argument_ref=make_argument_ref("some_tuple_none", 1),
        expected_is_result=False,
        expected_root_argument_name="some_tuple_none",
        expected_repr="some_tuple_none[1]",
        expected_get=[(None, sc(ArgumentContext("some_tuple_none"), IndexContext(1)))],
    ),
    ArgumentRefTest(
        name="tuple_missing_element",
        argument_ref=make_argument_ref("some_tuple", 2),
        expected_is_result=False,
        expected_root_argument_name="some_tuple",
        expected_repr="some_tuple[2]",
        expected_error="""
Unable to resolve argument / missing argument.
  Fake test error context.
    Argument: some_tuple
      Index: [2]
""",
    ),
    ArgumentRefTest(
        name="not_a_class",
        argument_ref=make_argument_ref(RESULT_TOKEN, 1, "a"),
        expected_is_result=True,
        expected_root_argument_name="return",
        expected_repr="return[1].a",
        expected_error="""
Unable to resolve argument / missing argument.
  Fake test error context.
    Argument: return
      Index: [1]
        Attribute: .a
""",
    ),
    ArgumentRefTest(
        name="all_elements__keys",
        argument_ref=make_argument_ref("many", all_ref, 1, keys_ref),
        expected_is_result=False,
        expected_root_argument_name="many",
        expected_repr="many[all][1].keys()",
        expected_get=[
            (
                "a",
                sc(
                    sc(
                        sc(ArgumentContext("many"), IndexContext(0)),
                        IndexContext(1),
                    ),
                    MappingKeyContext("a"),
                ),
            ),
            (
                "b",
                sc(
                    sc(
                        sc(ArgumentContext("many"), IndexContext(0)),
                        IndexContext(1),
                    ),
                    MappingKeyContext("b"),
                ),
            ),
            (None, sc(ArgumentContext("many"), IndexContext(1))),
            (
                None,
                sc(
                    sc(
                        sc(ArgumentContext("many"), IndexContext(2)),
                        IndexContext(1),
                    ),
                    MappingKeyContext(None),
                ),
            ),
            (
                "d",
                sc(
                    sc(
                        sc(ArgumentContext("many"), IndexContext(2)),
                        IndexContext(1),
                    ),
                    MappingKeyContext("d"),
                ),
            ),
        ],
    ),
    ArgumentRefTest(
        name="all_elements__values",
        argument_ref=make_argument_ref("many", all_ref, 1, values_ref, "many", all_ref),
        expected_is_result=False,
        expected_root_argument_name="many",
        expected_repr="many[all][1].values().many[all]",
        expected_get=[
            (
                111,
                sc(
                    sc(
                        sc(
                            sc(
                                sc(ArgumentContext("many"), IndexContext(0)),
                                IndexContext(1),
                            ),
                            MappingValueContext("a"),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(0),
                ),
            ),
            (
                112,
                sc(
                    sc(
                        sc(
                            sc(
                                sc(ArgumentContext("many"), IndexContext(0)),
                                IndexContext(1),
                            ),
                            MappingValueContext("a"),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(1),
                ),
            ),
            (
                121,
                sc(
                    sc(
                        sc(
                            sc(
                                sc(ArgumentContext("many"), IndexContext(0)),
                                IndexContext(1),
                            ),
                            MappingValueContext("b"),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(0),
                ),
            ),
            (
                122,
                sc(
                    sc(
                        sc(
                            sc(
                                sc(ArgumentContext("many"), IndexContext(0)),
                                IndexContext(1),
                            ),
                            MappingValueContext("b"),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(1),
                ),
            ),
            (None, sc(ArgumentContext("many"), IndexContext(1))),
            (
                None,
                sc(
                    sc(
                        sc(
                            sc(sc(ArgumentContext("many"), IndexContext(2)), IndexContext(1)),
                            MappingValueContext(None),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(0),
                ),
            ),
            (
                221,
                sc(
                    sc(
                        sc(
                            sc(
                                sc(ArgumentContext("many"), IndexContext(2)),
                                IndexContext(1),
                            ),
                            MappingValueContext(None),
                        ),
                        AttributeContext("many"),
                    ),
                    IndexContext(1),
                ),
            ),
            (
                None,
                sc(
                    sc(sc(ArgumentContext("many"), IndexContext(2)), IndexContext(1)),
                    MappingValueContext("d"),
                ),
            ),
        ],
    ),
]


@pytest.mark.parametrize("test", TESTS, ids=name)
def test_parse_argument_ref__is_result(test: ArgumentRefTest) -> None:
    assert test.expected_is_result == test.argument_ref.is_result


@pytest.mark.parametrize("test", TESTS, ids=name)
def test_parse_argument_ref__root_argument_name(test: ArgumentRefTest) -> None:
    assert test.expected_root_argument_name == test.argument_ref.root_argument_name


@pytest.mark.parametrize("test", TESTS, ids=name)
def test_parse_argument_ref__repr(test: ArgumentRefTest) -> None:
    assert test.expected_repr == repr(test.argument_ref)


@pytest.mark.parametrize("test", [test for test in TESTS if test.expected_error is None], ids=name)
def test_parse_argument_ref__get(test: ArgumentRefTest) -> None:
    values = test.argument_ref.get(ARG_MAP, TestContext())
    assert test.expected_get == values


@pytest.mark.parametrize(
    "test", [test for test in TESTS if test.expected_error is not None], ids=name
)
def test_parse_argument_ref__error(test: ArgumentRefTest) -> None:
    with pytest.raises(ArgumentReferenceError) as e:
        test.argument_ref.get(ARG_MAP, TestContext())
    (message,) = e.value.args
    assert test.expected_error == message
