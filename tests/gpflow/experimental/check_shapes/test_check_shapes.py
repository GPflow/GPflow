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
from typing import Tuple

import pytest

from gpflow.base import TensorType
from gpflow.experimental.check_shapes import ShapeMismatchError, check_shapes

from .utils import t


def test_check_shapes__constant() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4)

    # Don't crash...
    f(t(2, 3), t(2, 4))
    f(t(None, 3), t(2, 4))
    f(t(2, None), t(2, 4))
    f(t(2, 3), t(None, 4))


def test_check_shapes__constant__bad_input() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__constant__bad_return() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_dim() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4)

    # Don't crash...
    f(t(2, 3), t(2, 4))
    f(t(None, 3), t(2, 4))
    f(t(2, None), t(2, 4))
    f(t(2, 3), t(None, 4))


def test_check_shapes__var_dim__bad_input() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__var_dim__bad_return() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_rank() -> None:
    @check_shapes(
        ("a", ["ds..."]),
        ("b", ["ds...", "d1"]),
        ("return", ["ds...", "d1", "d2"]),
    )
    def f(a: TensorType, b: TensorType, leading_dims: int) -> TensorType:
        output_shape = leading_dims * (2,) + (3, 4)
        return t(*output_shape)

    # Don't crash...
    f(t(), t(3), leading_dims=0)
    f(t(2), t(2, 3), leading_dims=1)
    f(t(2, 2), t(2, 2, 3), leading_dims=2)
    f(t(None, 2), t(2, 2, 3), leading_dims=2)
    f(t(2, None), t(2, 2, 3), leading_dims=2)
    f(t(2, 2), t(None, 2, 3), leading_dims=2)
    f(t(2, 2), t(2, 2, None), leading_dims=2)


def test_check_shapes__var_rank__bad_input() -> None:
    @check_shapes(
        ("a", ["ds..."]),
        ("b", ["ds...", "d1"]),
        ("return", ["ds...", "d1", "d2"]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(1, 2, 3)

    with pytest.raises(ShapeMismatchError):
        f(t(1), t(1, 2 + 1))


def test_check_shapes__var_rank__bad_return() -> None:
    @check_shapes(
        ("a", ["ds..."]),
        ("b", ["ds...", "d1"]),
        ("return", ["ds...", "d1", "d2"]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(1, 2 + 1, 3)

    with pytest.raises(ShapeMismatchError):
        f(t(1), t(1, 2))


def test_check_shapes__scalar() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t()

    f(t(), t())  # Don't crash...


def test_check_shapes__scalar__bad_input() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t()

    with pytest.raises(ShapeMismatchError):
        f(t(1, 1), t())


def test_check_shapes__scalar__bad_return() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(1, 1)

    with pytest.raises(ShapeMismatchError):
        f(t(), t())


def test_check_shapes__invalid_argument() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: TensorType, b: TensorType) -> TensorType:
        return t(3, 4)

    with pytest.raises(TypeError):
        f(c=t(2, 3))  # type: ignore  # Intentionally invalid call.


def test_check_shapes__argument_refs() -> None:
    @dataclass
    class Input:
        ins: Tuple[TensorType, TensorType]

    @dataclass
    class Output:
        out: TensorType

    @check_shapes(
        ("x.ins[0]", ["a_batch...", 1]),
        ("x.ins[1]", ["b_batch...", 2]),
        ("return[0].out", ["a_batch...", 3]),
        ("return[1].out", ["b_batch...", 4]),
    )
    def f(x: Input) -> Tuple[Output, Output]:
        return (
            Output(out=t(2, 1, 3)),
            Output(out=t(1, 2, 4)),
        )

    f(Input(ins=(t(2, 1, 1), t(1, 2, 2))))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        f(Input(ins=(t(2, 1, 1), t(1, 1, 2))))
