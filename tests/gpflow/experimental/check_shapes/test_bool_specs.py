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
from typing import Any, Mapping, Tuple

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
                {"foo": [3]},
                (
                    True,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext([3]),
                    ),
                ),
            ),
            (
                {"foo": []},
                (
                    False,
                    StackContext(
                        ArgumentContext("foo"),
                        ObjectValueContext([]),
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
