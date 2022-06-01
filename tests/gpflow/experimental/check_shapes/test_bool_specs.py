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
Unit test for code for specifying and evaluating boolean expressions.
"""
from dataclasses import dataclass
from typing import Any, List, Mapping, Tuple

import pytest

from gpflow.experimental.check_shapes.bool_specs import ParsedBoolSpec
from gpflow.experimental.check_shapes.error_contexts import (
    ArgumentContext,
    ErrorContext,
    ObjectValueContext,
    ParallelContext,
    StackContext,
)

from .utils import TestContext, band, barg, bnot, bor

CONTEXT = TestContext()


@dataclass(frozen=True)
class BoolSpecTest:
    spec: ParsedBoolSpec
    expected_get: Tuple[Tuple[Mapping[str, Any], Tuple[bool, ErrorContext]], ...]
    expected_repr: str

    def __str__(self) -> str:
        return repr(self.spec).replace(" ", "_")


_NON_EMPTY_LIST = [3]
_EMPTY_LIST: List[int] = []


TESTS = [
    BoolSpecTest(
        spec=barg("foo"),
        expected_get=(
            (
                {"foo": True},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(True),
                    ),
                ),
            ),
            (
                {"foo": False},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(False),
                    ),
                ),
            ),
            (
                {"foo": 7},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(7),
                    ),
                ),
            ),
            (
                {"foo": 0},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(0),
                    ),
                ),
            ),
            (
                {"foo": _NON_EMPTY_LIST},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_NON_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": _EMPTY_LIST},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": None},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(None),
                    ),
                ),
            ),
        ),
        expected_repr="foo",
    ),
    BoolSpecTest(
        spec=barg("foo", "IS_NONE"),
        expected_get=(
            (
                {"foo": True},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(True),
                    ),
                ),
            ),
            (
                {"foo": False},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(False),
                    ),
                ),
            ),
            (
                {"foo": 7},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(7),
                    ),
                ),
            ),
            (
                {"foo": 0},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(0),
                    ),
                ),
            ),
            (
                {"foo": _NON_EMPTY_LIST},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_NON_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": _EMPTY_LIST},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": None},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(None),
                    ),
                ),
            ),
        ),
        expected_repr="foo is None",
    ),
    BoolSpecTest(
        spec=barg("foo", "IS_NOT_NONE"),
        expected_get=(
            (
                {"foo": True},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(True),
                    ),
                ),
            ),
            (
                {"foo": False},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(False),
                    ),
                ),
            ),
            (
                {"foo": 7},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(7),
                    ),
                ),
            ),
            (
                {"foo": 0},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(0),
                    ),
                ),
            ),
            (
                {"foo": _NON_EMPTY_LIST},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_NON_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": _EMPTY_LIST},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(_EMPTY_LIST),
                    ),
                ),
            ),
            (
                {"foo": None},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext(None),
                    ),
                ),
            ),
        ),
        expected_repr="foo is not None",
    ),
    BoolSpecTest(
        spec=bor(barg("left"), barg("right")),
        expected_get=(
            (
                {"left": True, "right": True},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": True, "right": False},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": True},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": False},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="left or right",
    ),
    BoolSpecTest(
        spec=bor(barg("left", "IS_NONE"), barg("right", "IS_NOT_NONE")),
        expected_get=(
            (
                {"left": 1, "right": 1},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(1)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(1)),
                        )
                    ),
                ),
            ),
            (
                {"left": 1, "right": None},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(1)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(None)),
                        )
                    ),
                ),
            ),
            (
                {"left": None, "right": 1},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(None)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(1)),
                        )
                    ),
                ),
            ),
            (
                {"left": None, "right": None},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(None)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(None)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="(left is None) or (right is not None)",
    ),
    BoolSpecTest(
        spec=band(barg("left"), barg("right")),
        expected_get=(
            (
                {"left": True, "right": True},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": True, "right": False},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": True},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": False},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="left and right",
    ),
    BoolSpecTest(
        spec=band(barg("left", "IS_NOT_NONE"), barg("right", "IS_NONE")),
        expected_get=(
            (
                {"left": 1, "right": 1},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(1)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(1)),
                        )
                    ),
                ),
            ),
            (
                {"left": 1, "right": None},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(1)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(None)),
                        )
                    ),
                ),
            ),
            (
                {"left": None, "right": 1},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(None)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(1)),
                        )
                    ),
                ),
            ),
            (
                {"left": None, "right": None},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(None)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(None)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="(left is not None) and (right is None)",
    ),
    BoolSpecTest(
        spec=bnot(barg("right")),
        expected_get=(
            (
                {"right": True},
                (False, StackContext(ArgumentContext("right"), ObjectValueContext(True))),
            ),
            (
                {"right": False},
                (True, StackContext(ArgumentContext("right"), ObjectValueContext(False))),
            ),
        ),
        expected_repr="not right",
    ),
    BoolSpecTest(
        spec=bnot(barg("right", "IS_NONE")),
        expected_get=(
            (
                {"right": 1},
                (True, StackContext(ArgumentContext("right"), ObjectValueContext(1))),
            ),
            (
                {"right": None},
                (False, StackContext(ArgumentContext("right"), ObjectValueContext(None))),
            ),
        ),
        expected_repr="not (right is None)",
    ),
    BoolSpecTest(
        spec=bnot(barg("right", "IS_NOT_NONE")),
        expected_get=(
            (
                {"right": 1},
                (False, StackContext(ArgumentContext("right"), ObjectValueContext(1))),
            ),
            (
                {"right": None},
                (True, StackContext(ArgumentContext("right"), ObjectValueContext(None))),
            ),
        ),
        expected_repr="not (right is not None)",
    ),
    BoolSpecTest(
        spec=bor(bnot(barg("left")), bnot(barg("right"))),
        expected_get=(
            (
                {"left": True, "right": True},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": True, "right": False},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": True},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": False},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="(not left) or (not right)",
    ),
    BoolSpecTest(
        spec=band(bnot(barg("left")), bnot(barg("right"))),
        expected_get=(
            (
                {"left": True, "right": True},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": True, "right": False},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(True)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": True},
                (
                    False,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(True)),
                        )
                    ),
                ),
            ),
            (
                {"left": False, "right": False},
                (
                    True,
                    ParallelContext(
                        (
                            StackContext(ArgumentContext("left"), ObjectValueContext(False)),
                            StackContext(ArgumentContext("right"), ObjectValueContext(False)),
                        )
                    ),
                ),
            ),
        ),
        expected_repr="(not left) and (not right)",
    ),
    BoolSpecTest(
        spec=bnot(bnot(barg("right"))),
        expected_get=(
            (
                {"right": True},
                (True, StackContext(ArgumentContext("right"), ObjectValueContext(True))),
            ),
            (
                {"right": False},
                (False, StackContext(ArgumentContext("right"), ObjectValueContext(False))),
            ),
        ),
        expected_repr="not (not right)",
    ),
]


@pytest.mark.parametrize("test", TESTS, ids=str)
def test_bool_spec__get(test: BoolSpecTest) -> None:
    for arg_map, expected in test.expected_get:
        assert expected == test.spec.get(arg_map, CONTEXT)


@pytest.mark.parametrize("test", TESTS, ids=str)
def test_bool_spec__repr(test: BoolSpecTest) -> None:
    assert test.expected_repr == repr(test.spec)
