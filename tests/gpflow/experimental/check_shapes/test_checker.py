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

# pylint: disable=unused-argument  # Bunch of fake functions below has unused arguments.

import sys
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import pytest

from gpflow.experimental.check_shapes import ShapeChecker, disable_check_shapes
from gpflow.experimental.check_shapes.checker import TensorSpecLike
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError, VariableTypeError

from .utils import TestContext, current_line, make_shape_spec, make_tensor_spec, t, t_unk


@dataclass(frozen=True)
class ShapeCheckerTest:
    name: str
    checks: Sequence[Tuple[Any, TensorSpecLike]]
    good: bool


def name(test: ShapeCheckerTest) -> str:
    return test.name


TESTS = [
    ShapeCheckerTest("scalar", [(t(), "[]")], True),
    ShapeCheckerTest("scalar_unknown_tensor", [(t_unk(), "[]")], True),
    ShapeCheckerTest("scalar_bad", [(t(2), "[]")], False),
    ShapeCheckerTest("constant", [(t(2, 3), "[2, 3]")], True),
    ShapeCheckerTest("constant_unknown_dim_1", [(t(None, 3), "[2, 3]")], True),
    ShapeCheckerTest("constant_unknown_dim_2", [(t(2, None), "[2, 3]")], True),
    ShapeCheckerTest("constant_unknown_tensor", [(t_unk(), "[2, 3]")], True),
    ShapeCheckerTest("constant_bad", [(t(2, 3), "[4, 3]")], False),
    ShapeCheckerTest("constant_unknown_dim_1_bad", [(t(None, 3), "[2, 2]")], False),
    ShapeCheckerTest("constant_unknown_dim_2_bad", [(t(2, None), "[3, 3]")], False),
    ShapeCheckerTest(
        "var_dim",
        [
            (t(2, 3), "[d1, d2]"),
            (t(2, 4), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_unknown_dim_1",
        [
            (t(None, 3), "[d1, d2]"),
            (t(2, 4), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_unknown_dim_2",
        [
            (t(2, None), "[d1, d2]"),
            (t(2, 4), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_unknown_dim_3",
        [
            (t(2, 3), "[d1, d2]"),
            (t(2, None), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_unknown_tensor_1",
        [
            (t_unk(), "[d1, d2]"),
            (t(2, 4), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_unknown_tensor_2",
        [
            (t(2, 3), "[d1, d2]"),
            (t_unk(), "[d1, d3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_bad",
        [
            (t(2, 3), "[d1, d2]"),
            (t(3, 4), "[d1, d3]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_dim_reuse",
        [
            (t(3, 3), "[d1, d1]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_dim_reuse_bad",
        [
            (t(3, 4), "[d1, d1]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_empty",
        [
            (t(), "[ds...]"),
            (t(2), "[d1, ds...]"),
            (t(2, 9), "[d1, ds..., d2]"),
            (t(9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_single",
        [
            (t(5), "[ds...]"),
            (t(2, 5), "[d1, ds...]"),
            (t(2, 5, 9), "[d1, ds..., d2]"),
            (t(5, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multiple",
        [
            (t(5, 6), "[ds...]"),
            (t(2, 5, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_1",
        [
            (t(1, 3, 6, 7, 5), "[d1, ds2..., d3, ds4..., d5]"),
            (t(), "[ds2...]"),
            (t(6, 7), "[ds4...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_2",
        [
            (t(1, 3, 6, 7, 5), "[d1, ds2..., d3, ds4..., d5]"),
            (t(6, 7), "[ds4...]"),
            (t(), "[ds2...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_3",
        [
            (t(), "[ds2...]"),
            (t(1, 3, 6, 7, 5), "[d1, ds2..., d3, ds4..., d5]"),
            (t(6, 7), "[ds4...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_4",
        [
            (t(), "[ds2...]"),
            (t(6, 7), "[ds4...]"),
            (t(1, 3, 6, 7, 5), "[d1, ds2..., d3, ds4..., d5]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_5",
        [
            (t(1, 2, 1, 2), "[ds..., ds...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_6",
        [
            (t(1, 2, 3, 4, 5, 6), "[1, ..., 3, ..., 6]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_multi_broadcast",
        [
            (t(2, 1, 1, 2, 3), "[broadcast ds..., ds...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_unknown_dim_1",
        [
            (t(None, 6), "[ds...]"),
            (t(2, 5, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_unknown_dim_2",
        [
            (t(5, 6), "[ds...]"),
            (t(None, 5, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_unknown_dim_3",
        [
            (t(5, 6), "[ds...]"),
            (t(2, 5, 6), "[d1, ds...]"),
            (t(2, 5, None, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_unknown_tensor_1",
        [
            (t_unk(), "[ds...]"),
            (t(2, 5, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_unknown_tensor_2",
        [
            (t(5, 6), "[ds...]"),
            (t(2, 5, 6), "[d1, ds...]"),
            (t_unk(), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "var_rank_bad_1",
        [
            (t(5, 6), "[ds...]"),
            (t(2, 3, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_2",
        [
            (t(5, 6), "[ds...]"),
            (t(3, 5, 6), "[d1, ds...]"),
            (t(2, 5, 6, 9), "[d1, ds..., d2]"),
            (t(5, 6, 9), "[ds..., d2]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_3",
        [(t(2), "[d1, ds..., d2]")],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_4",
        [
            (t(1, 2, 3), "[ds...]"),
            (t(2, 3), "[ds...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_5",
        [
            (t(2, 3), "[ds...]"),
            (t(1, 2, 3), "[ds...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_6",
        [
            (t(1, 4, 3), "[ds1..., ds2...]"),
            (t(1, 2), "[ds1...]"),
            (t(3), "[ds2...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_7",
        [
            (t(1, 2, 1, 2, 3), "[ds..., ds...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_8",
        [
            (t(1, 2, 1, 3), "[ds..., ds...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_9",
        [
            (t(7, 2, 3, 4, 5, 6), "[1, ..., 3, ..., 6]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_10",
        [
            (t(1, 2, 3, 4, 7), "[1, ..., 3, ..., 5]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "var_rank_bad_11",
        [
            (t(2, 1, 1, 2, 3), "[broadcast ds..., ds...]"),
            (t(1, 1, 2, 3), "[ds...]"),
        ],
        False,
    ),
    ShapeCheckerTest("anonymous_dot", [(t(2, 3), "[., 3]")], True),
    ShapeCheckerTest("anonymous_None", [(t(2, 3), "[2, None]")], True),
    ShapeCheckerTest("anonymous_ellipsis", [(t(2, 3), "[..., 3]")], True),
    ShapeCheckerTest("anonymous_star", [(t(2, 3), "[2, *]")], True),
    ShapeCheckerTest("anonymous_dot_unknown_dim", [(t(None, 3), "[., 3]")], True),
    ShapeCheckerTest("anonymous_dot_unknown_tensor", [(t_unk(), "[., 3]")], True),
    ShapeCheckerTest(
        "anonymous_dot_different_values",
        [
            (t(4, 2), "[., 2]"),
            (t(5, 3), "[., 3]"),
        ],
        True,
    ),
    ShapeCheckerTest("anonymous_ellipsis_unknown_dim", [(t(None, 3), "[..., 3]")], True),
    ShapeCheckerTest("anonymous_ellipsis_unknown_tensor", [(t_unk(), "[..., 3]")], True),
    ShapeCheckerTest(
        "anonymous_ellipsis_different_values",
        [
            (t(1, 3, 2), "[..., 2]"),
            (t(4, 3), "[..., 3]"),
        ],
        True,
    ),
    ShapeCheckerTest("anonymous_dot_bad_too_long", [(t(1, 2, 3), "[., 3]")], False),
    ShapeCheckerTest("anonymous_dot_bad_too_short", [(t(3), "[., 3]")], False),
    ShapeCheckerTest("anonymous_ellipsis_too_short", [(t(), "[..., 3]")], False),
    ShapeCheckerTest(
        "broadcast_constant",
        [
            (t(1, 3), "[broadcast 2, broadcast 3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_constant_short_1",
        [
            (t(3), "[broadcast 2, broadcast 3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_constant_short_2",
        [
            (t(), "[broadcast 2, broadcast 3]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_constant_bad",
        [
            (t(1, 4), "[broadcast 2, broadcast 3]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_neither_1",
        [
            (t(2, 3, 4), "[broadcast a, b, broadcast c]"),
            (t(2, 3, 4), "[a, broadcast b, broadcast c]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_broadcast_1",
        [
            (t(1, 3, 1, 5), "[broadcast a, b, broadcast c, broadcast d]"),
            (t(2, 1, 4, 1), "[a, broadcast b, broadcast c, broadcast d]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_both_1",
        [
            (t(1, 1, 1), "[broadcast a, b, broadcast c]"),
            (t(1, 1, 1), "[a, broadcast b, broadcast c]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_mismatch_1",
        [
            (t(2), "[broadcast a]"),
            (t(3), "[a]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_mismatch_2",
        [
            (t(2), "[a]"),
            (t(3), "[broadcast a]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_first",
        [
            (t(1, 1, 3), "[broadcast a...]"),
            (t(1, 2, 3), "[a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_second",
        [
            (t(1, 2, 3), "[a...]"),
            (t(1, 1, 3), "[broadcast a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_short_first",
        [
            (t(1, 1, 3), "[broadcast a...]"),
            (t(4, 4, 1, 2, 3), "[a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_short_second",
        [
            (t(4, 4, 1, 2, 3), "[a...]"),
            (t(1, 1, 3), "[broadcast a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_empty_first",
        [
            (t(), "[broadcast a...]"),
            (t(1, 2, 3), "[a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_empty_second",
        [
            (t(1, 2, 3), "[a...]"),
            (t(), "[broadcast a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_mismatch_first",
        [
            (t(1, 4, 3), "[broadcast a...]"),
            (t(1, 2, 3), "[a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_mismatch_second",
        [
            (t(1, 2, 3), "[a...]"),
            (t(1, 4, 3), "[broadcast a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_long_first",
        [
            (t(1, 2, 3), "[broadcast a...]"),
            (t(2, 3), "[a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_long_second",
        [
            (t(2, 3), "[a...]"),
            (t(1, 2, 3), "[broadcast a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_double",
        [
            (t(3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(3, 4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 3, 4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 3, 4), "[a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_double_bad",
        [
            (t(3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(3, 4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 3, 4, 2, 3, 4), "[broadcast a..., broadcast a...]"),
            (t(2, 4, 4), "[a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_1",
        [
            (t(2, 3, 4, 1, 3, 1), "[a..., broadcast a...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_1_bad",
        [
            (t(2, 3, 4, 3, 4), "[a..., broadcast a...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_2",
        [
            (t(2, 3, 4), "[a, broadcast b...]"),
            (t(3, 4), "[b...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_2_bad",
        [
            (t(2, 3, 4), "[a, broadcast b...]"),
            (t(2, 3, 4), "[b...]"),
        ],
        False,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_3",
        [
            (t(2, 3, 4, 5, 6, 7), "[broadcast a, broadcast b..., c, d...]"),
            (t(6, 7), "[d...]"),
            (t(3, 4), "[b...]"),
        ],
        True,
    ),
    ShapeCheckerTest(
        "broadcast_variable_rank_trailing_3_bad",
        [
            (t(2, 3, 4, 5, 6, 7), "[broadcast a, broadcast b..., c, d...]"),
            (t(3), "[a]"),
            (t(6, 7), "[d...]"),
            (t(3, 4), "[b...]"),
        ],
        False,
    ),
    ShapeCheckerTest("None", [(None, "[2, 3]")], True),
    ShapeCheckerTest(
        "parsed_tensor_spec",
        [
            (t(1, 2, 3), make_tensor_spec(make_shape_spec(1, 2, 3), None)),
        ],
        True,
    ),
    ShapeCheckerTest(
        "tuple_tensor_spec",
        [
            (t(1, 2, None), (None, 2, 3)),
            (t(1, 2, 3), None),
        ],
        True,
    ),
]
POSITIVE_TESTS = [test for test in TESTS if test.good]
NEGATIVE_TESTS = [test for test in TESTS if not test.good]


@pytest.mark.parametrize("test", POSITIVE_TESTS, ids=name)
def test_shape_checker__check_shape__positive(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    for shaped, shape_spec in test.checks:
        assert shaped is checker.check_shape(shaped, shape_spec)  # Don't crash.


@pytest.mark.parametrize("test", NEGATIVE_TESTS, ids=name)
def test_shape_checker__check_shape__negative(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    with pytest.raises(ShapeMismatchError):
        for shaped, shape_spec in test.checks:
            checker.check_shape(shaped, shape_spec)


@pytest.mark.parametrize("test", TESTS, ids=name)
def test_shape_checker__check_shape__disable(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    for shaped, shape_spec in test.checks:
        with disable_check_shapes():
            checker.check_shape(shaped, shape_spec)


@pytest.mark.parametrize("test", POSITIVE_TESTS, ids=name)
def test_shape_checker__check_shapes__positive(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    checker.check_shapes(test.checks)  # Don't crash.


@pytest.mark.parametrize("test", NEGATIVE_TESTS, ids=name)
def test_shape_checker__check_shapes__negative(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    with pytest.raises(ShapeMismatchError):
        checker.check_shapes(test.checks)


@pytest.mark.parametrize("test", TESTS, ids=name)
def test_shape_checker__check_shapes__disable(test: ShapeCheckerTest) -> None:
    checker = ShapeChecker()
    with disable_check_shapes():
        checker.check_shapes(test.checks)


def test_shape_checker__error_message() -> None:
    # Here we're just testing that error message formatting is wired together sanely. For more
    # thorough tests of error formatting, see test_error_contexts.py and test_exceptions.py
    checker = ShapeChecker()

    test_context = TestContext("Test context")

    call_1_line = current_line() + 1
    checker.check_shape(t(2, 4), "[d2, d4]")

    call_2_line = current_line() + 1
    checker.check_shape(t(2, 3), "[d2, d3]  # Some note.")

    checker.check_shape(t(1, 4), "[d1, d4]", test_context)

    # How line numbers are determined has changed with Python versions. We can remove this if we
    # bump the minimum version of Python to >= 3.8:
    call_offset = 4 if sys.version_info < (3, 8) else 1

    call_4_line = current_line() + call_offset
    checker.check_shapes(
        [
            (t(3, 4), "[d3, d4]"),
            (t(4, 5), "[d4, d5]  # Another note.", test_context),
        ]
    )

    with pytest.raises(ShapeMismatchError) as e:
        call_5_line = current_line() + call_offset
        checker.check_shapes(
            [
                (t(2, 3, 1), "[d1, d2, d3]"),
                (t(5, 4), "[d4, d5]"),
            ]
        )

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  check_shape called at:  {__file__}:{call_1_line}
    Expected: [d2, d4]
    Actual:   [2, 4]
  check_shape called at:  {__file__}:{call_2_line}
    Note:     Some note.
    Expected: [d2, d3]
    Actual:   [2, 3]
  Test context
    Expected: [d1, d4]
    Actual:   [1, 4]
    Note:     Another note.
    Expected: [d4, d5]
    Actual:   [4, 5]
  check_shapes called at: {__file__}:{call_4_line}
    Index: [0]
      Expected: [d3, d4]
      Actual:   [3, 4]
  check_shapes called at: {__file__}:{call_5_line}
    Index: [0]
      Expected: [d1, d2, d3]
      Actual:   [2, 3, 1]
    Index: [1]
      Expected: [d4, d5]
      Actual:   [5, 4]
"""
        == message
    )


def test_shape_checker__error_message__variable_type() -> None:
    checker = ShapeChecker()

    call_1_line = current_line() + 1
    checker.check_shape(t(1), "[d1]")

    call_2_line = current_line() + 1
    checker.check_shape(t(2, 2), "[d2...]")

    # How line numbers are determined has changed with Python versions. We can remove this if we
    # bump the minimum version of Python to >= 3.8:
    call_offset = 4 if sys.version_info < (3, 8) else 1

    with pytest.raises(VariableTypeError) as e:
        call_3_line = current_line() + call_offset
        checker.check_shapes(
            [
                (t(2), "[d2]"),
                (t(1, 1), "[d1...]"),
            ]
        )

    (message,) = e.value.args
    assert (
        f"""
Cannot use the same variable to bind both a single dimension and a variable number of dimensions.
  Variable: d1
    check_shape called at:  {__file__}:{call_1_line}
      Specification: [d1]
    check_shapes called at: {__file__}:{call_3_line}
      Index: [1]
        Specification: [d1...]
  Variable: d2
    check_shape called at:  {__file__}:{call_2_line}
      Specification: [d2...]
    check_shapes called at: {__file__}:{call_3_line}
      Index: [0]
        Specification: [d2]
"""
        == message
    )
