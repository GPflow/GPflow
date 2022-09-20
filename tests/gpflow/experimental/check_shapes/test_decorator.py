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

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import pytest

from gpflow.experimental.check_shapes import (
    check_shape,
    check_shapes,
    disable_check_shapes,
    get_check_shapes,
)
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError

from .utils import TestShaped, current_line, t, t_unk


def get_shape(x: TestShaped) -> Tuple[Optional[int], ...]:
    shape = x.test_shape
    assert shape is not None
    return shape


def test_check_shapes__constant() -> None:
    @check_shapes(
        "a: [2, 3]",
        "b: [2, 4]",
        "return: [3, 4]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    # Don't crash...
    f(t(2, 3), t(2, 4))
    f(t(None, 3), t(2, 4))
    f(t(2, None), t(2, 4))
    f(t(2, 3), t(None, 4))
    f(t_unk(), t(2, 4))
    f(t(2, 3), t_unk())


def test_check_shapes__constant__bad_input() -> None:
    @check_shapes(
        "a: [2, 3]",
        "b: [2, 4]",
        "return: [3, 4]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__constant__bad_return() -> None:
    @check_shapes(
        "a: [2, 3]",
        "b: [2, 4]",
        "return: [3, 4]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_dim() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    # Don't crash...
    f(t(2, 3), t(2, 4))
    f(t(None, 3), t(2, 4))
    f(t(2, None), t(2, 4))
    f(t(2, 3), t(None, 4))
    f(t(2, 3), t(2, None))
    f(t_unk(), t(2, 4))
    f(t(2, 3), t_unk())


def test_check_shapes__var_dim__bad_input() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__var_dim__bad_return() -> None:
    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_rank() -> None:
    @check_shapes(
        "a: [ds...]",
        "b: [ds..., d1]",
        "c: [d1, ds..., d2]",
        "d: [d1, ds...]",
        "return: [ds..., d1, d2]",
    )
    def f(
        a: TestShaped, b: TestShaped, c: TestShaped, d: TestShaped, leading_dims: int
    ) -> TestShaped:
        output_shape = leading_dims * (2,) + (3, 4)
        return t(*output_shape)

    # Don't crash...
    f(t(), t(3), t(3, 4), t(3), leading_dims=0)
    f(t(2), t(2, 3), t(3, 2, 4), t(3, 2), leading_dims=1)
    f(t(2, 2), t(2, 2, 3), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(None, 2), t(2, 2, 3), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(2, None), t(2, 2, 3), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(2, 2), t(None, 2, 3), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(2, 2), t(2, 2, None), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t_unk(), t(2, 2, 3), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(2, 2), t_unk(), t(3, 2, 2, 4), t(3, 2, 2), leading_dims=2)
    f(t(2, 2), t(2, 2, 3), t_unk(), t(3, 2, 2), leading_dims=2)
    f(t(2, 2), t(2, 2, 3), t(3, 2, 2, 4), t_unk(), leading_dims=2)


def test_check_shapes__var_rank__bad_input() -> None:
    @check_shapes(
        "a: [ds...]",
        "b: [ds..., d1]",
        "c: [d1, ds..., d2]",
        "d: [d1, ds...]",
        "return: [ds..., d1, d2]",
    )
    def f(a: TestShaped, b: TestShaped, c: TestShaped, d: TestShaped) -> TestShaped:
        return t(1, 2, 3)

    with pytest.raises(ShapeMismatchError):
        f(t(2), t(2, 3), t(3, 2 + 1, 4), t(3, 2))


def test_check_shapes__var_rank__bad_return() -> None:
    @check_shapes(
        "a: [ds...]",
        "b: [ds..., d1]",
        "c: [d1, ds..., d2]",
        "d: [d1, ds...]",
        "return: [ds..., d1, d2]",
    )
    def f(a: TestShaped, b: TestShaped, c: TestShaped, d: TestShaped) -> TestShaped:
        return t(2, 3 + 1, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2), t(2, 3), t(3, 2, 4), t(3, 2))


def test_check_shapes__anonymous() -> None:
    @check_shapes(
        "a: [., d1]",
        "b: [None, d2]",
        "c: [..., d1]",
        "d: [*, d2]",
        "return: [..., d1, d2]",
    )
    def f(a: TestShaped, b: TestShaped, c: TestShaped, d: TestShaped) -> TestShaped:
        return t(
            *get_shape(c)[:-1],
            get_shape(a)[-1],
            get_shape(b)[-1],
        )

    f(t(1, 2), t(1, 3), t(2), t(3))
    f(t(1, 2), t(1, 3), t(1, 2), t(1, 3))
    f(t(1, 2), t(1, 3), t(1, 1, 2), t(1, 1, 3))
    f(t(None, 2), t(1, 3), t(2), t(3))
    f(t(1, None), t(1, 3), t(2), t(3))
    f(t(1, 2), t(None, 3), t(2), t(3))
    f(t(1, 2), t(1, None), t(2), t(3))
    f(t(1, 2), t(1, 3), t(None), t(3))
    f(t(1, 2), t(1, 3), t(2), t(None))


def test_check_shapes__anonymous__bad_imput() -> None:
    @check_shapes(
        "a: [., d1]",
        "b: [None, d2]",
        "c: [..., d1]",
        "d: [*, d2]",
        "return: [..., d1, d2]",
    )
    def f(a: TestShaped, b: TestShaped, c: TestShaped, d: TestShaped) -> TestShaped:
        return t(
            *get_shape(c)[:-1],
            get_shape(a)[-1],
            get_shape(b)[-1],
        )

    with pytest.raises(ShapeMismatchError):
        f(t(2), t(1, 3), t(2), t(3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 1, 2), t(1, 3), t(2), t(3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2), t(3), t(2), t(3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2), t(1, 1, 3), t(2), t(3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2), t(1, 3), t(), t(3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2), t(1, 3), t(2), t())


def test_check_shapes__broadcast_constant() -> None:
    @check_shapes(
        "a: [broadcast 2, broadcast 3]",
    )
    def f(a: TestShaped) -> None:
        pass

    f(t(1, 1))
    f(t(1, 3))
    f(t(2, 1))
    f(t(2, 3))
    f(t(None, 1))
    f(t(None, 3))
    f(t(1, None))
    f(t(2, None))
    f(t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(3, 3))


def test_check_shapes__broadcast_first() -> None:
    @check_shapes(
        "a: [broadcast d1, broadcast d2, broadcast d3]",
        "b: [d1, d2, d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 1, 3), t(1, 2, 3))
    f(t(None, 1, 3), t(1, 2, 3))
    f(t(1, None, 3), t(1, 2, 3))
    f(t(1, 1, None), t(1, 2, 3))
    f(t(1, 1, 3), t(None, 2, 3))
    f(t(1, 1, 3), t(1, None, 3))
    f(t(1, 1, 3), t(1, 2, None))
    f(t_unk(), t(1, 2, 3))
    f(t(1, 1, 3), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3), t(1, 4, 3))


def test_check_shapes__broadcast_second() -> None:
    @check_shapes(
        "a: [d1, d2, d3]",
        "b: [broadcast d1, broadcast d2, broadcast d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 2, 3), t(1, 1, 3))
    f(t(None, 2, 3), t(1, 1, 3))
    f(t(1, None, 3), t(1, 1, 3))
    f(t(1, 2, None), t(1, 1, 3))
    f(t(1, 2, 3), t(None, 1, 3))
    f(t(1, 2, 3), t(1, None, 3))
    f(t(1, 2, 3), t(1, 1, None))
    f(t_unk(), t(1, 1, 3))
    f(t(1, 2, 3), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3), t(1, 4, 3))


def test_check_shapes__broadcast_both() -> None:
    @check_shapes(
        "a: [broadcast d1, broadcast d2, broadcast d3, broadcast d4]",
        "b: [broadcast d1, broadcast d2, broadcast d3, broadcast d4]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 1, 3, 4), t(1, 2, 1, 4))
    f(t(None, 1, 3, 4), t(1, 2, 1, 4))
    f(t(1, None, 3, 4), t(1, 2, 1, 4))
    f(t(1, 1, None, 4), t(1, 2, 1, 4))
    f(t(1, 1, 3, None), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t(None, 2, 1, 4))
    f(t(1, 1, 3, 4), t(1, None, 1, 4))
    f(t(1, 1, 3, 4), t(1, 2, None, 4))
    f(t(1, 1, 3, 4), t(1, 2, 1, None))
    f(t_unk(), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3, 4), t(1, 2, 5, 4))


def test_check_shapes__broadcast_var_rank_first() -> None:
    @check_shapes(
        "a: [broadcast d...]",
        "b: [d...]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 1, 3), t(1, 2, 3))
    f(t(1), t(1, 2, 3))
    f(t(3), t(1, 2, 3))
    f(t(), t(1, 2, 3))
    f(t(None, 1, 3), t(1, 2, 3))
    f(t(1, None, 3), t(1, 2, 3))
    f(t(1, 1, None), t(1, 2, 3))
    f(t(1, 1, 3), t(None, 2, 3))
    f(t(1, 1, 3), t(1, None, 3))
    f(t(1, 1, 3), t(1, 2, None))
    f(t_unk(), t(1, 2, 3))
    f(t(1, 1, 3), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3), t(1, 4, 3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 1, 2, 3), t(1, 2, 3))


def test_check_shapes__broadcast_var_rank_second() -> None:
    @check_shapes(
        "a: [d...]",
        "b: [broadcast d...]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 2, 3), t(1, 1, 3))
    f(t(1, 2, 3), t(1))
    f(t(1, 2, 3), t(3))
    f(t(1, 2, 3), t())
    f(t(None, 2, 3), t(1, 1, 3))
    f(t(1, None, 3), t(1, 1, 3))
    f(t(1, 2, None), t(1, 1, 3))
    f(t(1, 2, 3), t(None, 1, 3))
    f(t(1, 2, 3), t(1, None, 3))
    f(t(1, 2, 3), t(1, 1, None))
    f(t_unk(), t(1, 1, 3))
    f(t(1, 2, 3), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3), t(1, 4, 3))

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3), t(1, 1, 2, 3))


def test_check_shapes__broadcast_var_rank_both() -> None:
    @check_shapes(
        "a: [broadcast d...]",
        "b: [broadcast d...]",
    )
    def f(a: TestShaped, b: TestShaped) -> None:
        pass

    f(t(1, 1, 3, 4), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t(1, 2, 1, 4))
    f(t(4), t(1, 2, 1, 4))
    f(t(1), t(1, 2, 1, 4))
    f(t(), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t(4))
    f(t(1, 1, 3, 4), t(1))
    f(t(1, 1, 3, 4), t())
    f(t(None, 1, 3, 4), t(1, 2, 1, 4))
    f(t(1, None, 3, 4), t(1, 2, 1, 4))
    f(t(1, 1, None, 4), t(1, 2, 1, 4))
    f(t(1, 1, 3, None), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t(None, 2, 1, 4))
    f(t(1, 1, 3, 4), t(1, None, 1, 4))
    f(t(1, 1, 3, 4), t(1, 2, None, 4))
    f(t(1, 1, 3, 4), t(1, 2, 1, None))
    f(t_unk(), t(1, 2, 1, 4))
    f(t(1, 1, 3, 4), t_unk())
    f(t_unk(), t_unk())

    with pytest.raises(ShapeMismatchError):
        f(t(1, 2, 3, 4), t(1, 2, 5, 4))


def test_check_shapes__scalar() -> None:
    @check_shapes(
        "a: []",
        "b: []",
        "return: []",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t()

    f(t(), t())  # Don't crash...


def test_check_shapes__scalar__bad_input() -> None:
    @check_shapes(
        "a: []",
        "b: []",
        "return: []",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t()

    with pytest.raises(ShapeMismatchError):
        f(t(1, 1), t())


def test_check_shapes__scalar__bad_return() -> None:
    @check_shapes(
        "a: []",
        "b: []",
        "return: []",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(1, 1)

    with pytest.raises(ShapeMismatchError):
        f(t(), t())


def test_check_shapes__invalid_argument() -> None:
    @check_shapes(
        "a: [2, 3]",
        "b: [2, 4]",
        "return: [3, 4]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    with pytest.raises(TypeError):
        # Linter disables, because we're intentionally making an invalid call.
        # pylint: disable=unexpected-keyword-arg,no-value-for-parameter
        f(c=t(2, 3))  # type: ignore[call-arg]


def test_check_shapes__argument_refs() -> None:
    @dataclass
    class Input:
        ins: Tuple[TestShaped, TestShaped]

    @dataclass
    class Output:
        out: TestShaped

    @check_shapes(
        "x.ins[0]: [a_batch..., 1]",
        "x.ins[1]: [b_batch..., 2]",
        "return[0].out: [a_batch..., 3]",
        "return[1].out: [b_batch..., 4]",
    )
    def f(x: Input) -> Tuple[Output, Output]:
        return (
            Output(out=t(2, 1, 3)),
            Output(out=t(1, 2, 4)),
        )

    f(Input(ins=(t(2, 1, 1), t(1, 2, 2))))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        f(Input(ins=(t(2, 1, 1), t(1, 1, 2))))


def test_check_shapes__argument_refs__collections() -> None:
    @check_shapes(
        "x.keys(): [., w]",
        "x.values(): [., h]",
        "return[all]: [w, h]",
    )
    def f(x: Mapping[TestShaped, TestShaped]) -> Sequence[TestShaped]:
        result = []
        for k, v in x.items():
            _, w = get_shape(k)
            _, h = get_shape(v)
            result.append(t(w, h))
        return result

    # Don't crash...
    f({})
    f(
        {
            t(1, 2): t(3, 4),
            t(5, 2): t(5, 4),
        }
    )

    with pytest.raises(ShapeMismatchError):
        f(
            {
                t(1, 2): t(3, 4),
                t(5, 3): t(5, 4),
            }
        )


def test_check_shapes__none() -> None:
    @dataclass
    class Input:
        ins: Optional[Tuple[Optional[TestShaped], ...]]

    @dataclass
    class Output:
        out: Optional[TestShaped]

    @check_shapes(
        "x.ins[0]: [1, 2]",
        "return[0].out: [3, 4]",
    )
    def f(x: Optional[Input]) -> Tuple[Output, ...]:
        return (Output(out=t(3, 4)),)

    # Don't crash...
    f(Input(ins=(t(1, 2),)))
    f(None)
    f(Input(ins=None))
    f(Input(ins=(None,)))


def test_check_shapes__condition() -> None:
    @check_shapes(
        "a: [1] if b1",
        "a: [2] if not b1 and b2",
        "a: [3] if not b1 and b2 is not None and not b2",
        "a: [4] if not b1 and b2 is None",
    )
    def f(a: TestShaped, b1: bool, b2: Optional[bool]) -> None:
        pass

    f(t(1), True, True)
    f(t(1), True, False)
    f(t(1), True, None)
    f(t(2), False, True)
    f(t(3), False, False)
    f(t(4), False, None)

    with pytest.raises(ShapeMismatchError):
        f(t(2), True, True)

    with pytest.raises(ShapeMismatchError):
        f(t(2), True, False)

    with pytest.raises(ShapeMismatchError):
        f(t(2), True, None)

    with pytest.raises(ShapeMismatchError):
        f(t(1), False, True)

    with pytest.raises(ShapeMismatchError):
        f(t(2), False, False)

    with pytest.raises(ShapeMismatchError):
        f(t(2), False, None)


def test_check_shapes__reuse() -> None:
    check_my_shapes = check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )

    @check_my_shapes
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    @check_my_shapes
    def g(a: TestShaped, b: TestShaped) -> TestShaped:
        return t(3, 4)

    assert get_check_shapes(f) is check_my_shapes
    assert get_check_shapes(g) is check_my_shapes

    # Don't crash...
    f(t(2, 3), t(2, 4))
    g(t(2, 3), t(2, 4))


def test_check_shapes__check_shape() -> None:

    def_line = current_line() + 3
    call_line = current_line() + 8

    @check_shapes(
        "a: [d1, d2]",
        "b: [d1, d3]",
        "return: [d2, d3]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        check_shape(t(4, 3), "[d3, d2]")
        return t(3, 4)

    # Don't crash...
    f(t(2, 3), t(2, 4))

    with pytest.raises(ShapeMismatchError) as e:
        f(t(2, 3), t(2, 5))

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  Function:              test_check_shapes__check_shape.<locals>.f
    Declared: {__file__}:{def_line}
    Argument: a
      Expected: [d1, d2]
      Actual:   [2, 3]
    Argument: b
      Expected: [d1, d3]
      Actual:   [2, 5]
  check_shape called at: {__file__}:{call_line}
    Expected: [d3, d2]
    Actual:   [4, 3]
"""
        == message
    )


def test_check_shapes__disable() -> None:
    def h(a: TestShaped, b: TestShaped) -> TestShaped:
        return a

    with disable_check_shapes():

        @check_shapes(
            "a: [d...]",
            "b: [d...]",
            "return: [d...]",
        )
        def f(a: TestShaped, b: TestShaped) -> TestShaped:
            return a

        f(t(2, 3), t(2, 4))  # Wrong shape, but checks disabled.

    f(t(2, 3), t(2, 4))  # Wrong shape, but checks were disable when function was created.
    get_check_shapes(f)(h)  # Don't crash.

    @check_shapes(
        "a: [d...]",
        "b: [d...]",
        "return: [d...]",
    )
    def g(a: TestShaped, b: TestShaped) -> TestShaped:
        return a

    with pytest.raises(ShapeMismatchError):
        g(t(2, 3), t(2, 4))
        get_check_shapes(g)(h)  # Don't crash.

    with disable_check_shapes():
        g(t(2, 3), t(2, 4))  # Wrong shape, but checks disabled.
        get_check_shapes(g)(h)  # Don't crash.

    with pytest.raises(ShapeMismatchError):
        g(t(2, 3), t(2, 4))
        get_check_shapes(g)(h)  # Don't crash.


def test_check_shapes__error_message() -> None:
    # Here we're just testing that error message formatting is wired together sanely. For more
    # thorough tests of error formatting, see test_error_contexts.py and test_exceptions.py

    def_line = current_line() + 2

    @check_shapes(
        "a: [d1, d2] if cond",
        "b: [d1, d3]  # Some note on b",
        "return: [d2, d3]",
        "# Some note on f",
        "# Some other note on f",
    )
    def f(a: TestShaped, b: TestShaped, cond: bool) -> TestShaped:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError) as e:
        f(t(2, 3), t(3, 4), True)

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  Function: test_check_shapes__error_message.<locals>.f
    Declared: {__file__}:{def_line}
    Note:     Some note on f
    Note:     Some other note on f
    Argument: a
      Condition: cond
        Argument: cond
          Value: True
      Expected: [d1, d2]
      Actual:   [2, 3]
    Argument: b
      Note:     Some note on b
      Expected: [d1, d3]
      Actual:   [3, 4]
"""
        == message
    )


def test_check_shapes__rewrites_docstring() -> None:
    # Here we're just testing that the rewrite is applied. For tests of formatting, see
    # test_parser.py

    @check_shapes(
        "a: [2, 3]",
        "b: [2, 4]",
        "return: [3, 4]",
    )
    def f(a: TestShaped, b: TestShaped) -> TestShaped:
        """
        A function for testing shape checking.

        :param a: The first parameter.
        :param b: The second parameter.
        :returns: The result.
        """
        return t(3, 4)

    assert (
        """
        A function for testing shape checking.

        :param a:
            * **a** has shape [2, 3].

            The first parameter.
        :param b:
            * **b** has shape [2, 4].

            The second parameter.
        :returns:
            * **return** has shape [3, 4].

            The result.
        """
        == f.__doc__
    )
