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
Unit tests for tool for checking the shapes of function using tf Tensors.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf

from gpflow.experimental.check_shapes import (
    ArgumentReferenceError,
    Shaped,
    ShapeMismatchError,
    check_shapes,
    inherit_check_shapes,
)


def t(*shape: Optional[int]) -> Shaped:
    mock_tensor = MagicMock()
    mock_tensor.shape = shape
    return mock_tensor


def test_check_shapes__constant() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
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
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__constant__bad_return() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_dim() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
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
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(3, 4)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2 + 1, 4))


def test_check_shapes__var_dim__bad_return() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(3, 4 + 1)

    with pytest.raises(ShapeMismatchError):
        f(t(2, 3), t(2, 4))


def test_check_shapes__var_rank() -> None:
    @check_shapes(
        ("a", ["ds..."]),
        ("b", ["ds...", "d1"]),
        ("return", ["ds...", "d1", "d2"]),
    )
    def f(a: Shaped, b: Shaped, leading_dims: int) -> Shaped:
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
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(1, 2, 3)

    with pytest.raises(ShapeMismatchError):
        f(t(1), t(1, 2 + 1))


def test_check_shapes__var_rank__bad_return() -> None:
    @check_shapes(
        ("a", ["ds..."]),
        ("b", ["ds...", "d1"]),
        ("return", ["ds...", "d1", "d2"]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(1, 2 + 1, 3)

    with pytest.raises(ShapeMismatchError):
        f(t(1), t(1, 2))


def test_check_shapes__scalar() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t()

    f(t(), t())  # Don't crash...


def test_check_shapes__scalar__bad_input() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t()

    with pytest.raises(ShapeMismatchError):
        f(t(1, 1), t())


def test_check_shapes__scalar__bad_return() -> None:
    @check_shapes(
        ("a", []),
        ("b", []),
        ("return", []),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(1, 1)

    with pytest.raises(ShapeMismatchError):
        f(t(), t())


def test_check_shapes__invalid_argument() -> None:
    @check_shapes(
        ("a", [2, 3]),
        ("b", [2, 4]),
        ("return", [3, 4]),
    )
    def f(a: Shaped, b: Shaped) -> Shaped:
        return t(3, 4)

    with pytest.raises(TypeError):
        f(c=t(2, 3))


def test_check_shapes__invalid_argument_name() -> None:
    with pytest.raises(AssertionError):

        @check_shapes(
            ("a", [2, 3]),
            ("#$%#$", [2, 4]),
            ("return", [3, 4]),
        )
        def f(a: Shaped, b: Shaped) -> Shaped:
            return t(3, 4)


def test_check_shapes__invalid_nested_argument_name() -> None:
    with pytest.raises(AssertionError):

        @check_shapes(
            ("a", [2, 3]),
            ("b#$%#$", [2, 4]),
            ("return", [3, 4]),
        )
        def f(a: Shaped, b: Shaped) -> Shaped:
            return t(3, 4)


def test_check_shapes__argument_refs() -> None:
    @dataclass
    class Input:
        ins: Tuple[Shaped, Shaped]

    @dataclass
    class Output:
        out: Shaped

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


def test_check_shapes__argument_refs__bad_attribute() -> None:
    @dataclass
    class Input:
        ins: Shaped

    @check_shapes(
        ("x.outs", ["a_batch...", 1]),
    )
    def f(x: Input) -> None:
        pass

    with pytest.raises(ArgumentReferenceError):
        f(Input(ins=t(3, 2, 1)))


def test_check_shapes__argument_refs__bad_index() -> None:
    @check_shapes(
        ("x[2]", ["a_batch...", 1]),
    )
    def f(x: Tuple[Shaped, Shaped]) -> None:
        pass

    with pytest.raises(ArgumentReferenceError):
        f((t(3, 2, 1), t(2, 3, 1)))


def test_check_shapes__numpy() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.zeros((3, 4))

    f(np.zeros((2, 3)), np.zeros((2, 4)))  # Don't crash...


def test_check_shapes__tensorflow() -> None:
    @check_shapes(
        ("a", ["d1", "d2"]),
        ("b", ["d1", "d3"]),
        ("return", ["d2", "d3"]),
    )
    def f(a: tf.Tensor, b: tf.Tensor) -> tf.Tensor:
        return tf.zeros((3, 4))

    f(tf.zeros((2, 3)), tf.zeros((2, 4)))  # Don't crash...


@pytest.mark.parametrize(
    "f_wrapper,loss_wrapper",
    [
        (
            lambda x: x,
            lambda x: x,
        ),
        (
            tf.function,
            tf.function,
        ),
        (
            tf.function(input_signature=[tf.TensorSpec(shape=[None], dtype=tf.float64)]),
            tf.function(input_signature=[]),
        ),
        (
            tf.function(input_signature=[tf.TensorSpec(shape=None, dtype=tf.float64)]),
            tf.function(input_signature=[]),
        ),
        (
            tf.function(experimental_relax_shapes=True),
            tf.function(experimental_relax_shapes=True),
        ),
    ],
)
def test_check_shapes__tensorflow_compilation(
    f_wrapper: Callable[[tf.Tensor], tf.Tensor], loss_wrapper: Callable[[], tf.Tensor]
) -> None:
    target = 0.5

    @f_wrapper
    @check_shapes(
        ("x", ["n"]),
        ("return", ["n"]),
    )
    def f(x: tf.Tensor) -> Tuple[tf.Tensor]:
        return (x - target) ** 2

    v = tf.Variable(np.linspace(0.0, 1.0))

    @loss_wrapper
    @check_shapes(
        ("return", [1]),
    )
    def loss() -> tf.Tensor:
        # keepdims is just to add an extra dimension to make the check more interesting.
        return tf.reduce_sum(f(v), keepdims=True)

    optimiser = tf.keras.optimizers.SGD(learning_rate=0.25)
    for _ in range(10):
        optimiser.minimize(loss, var_list=[v])

    np.testing.assert_allclose(target, v.numpy(), atol=0.01)


def test_inherit_check_shapes__defined_in_super_class() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: Shaped) -> Shaped:
            pass

    class MiddleClass(SuperClass):
        pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__overridden_with_checks() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: Shaped) -> Shaped:
            pass

    class MiddleClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__overridden_without_checks() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: Shaped) -> Shaped:
            pass

    class MiddleClass(SuperClass):
        def f(self, a: Shaped) -> Shaped:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__defined_in_middle_class() -> None:
    class SuperClass(ABC):
        pass

    class MiddleClass(SuperClass):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: Shaped) -> Shaped:
            pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


# In multiple inheratiance
# Missing


def test_inherit_check_shapes__multiple_inheritance() -> None:
    class Left(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: Shaped) -> Shaped:
            pass

    class Right(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [5]),
            ("return", [1]),
        )
        def g(self, a: Shaped) -> Shaped:
            pass

    class SubClass(Left, Right):
        @inherit_check_shapes
        def f(self, a: Shaped) -> Shaped:
            return t(1)

        @inherit_check_shapes
        def g(self, a: Shaped) -> Shaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...
    sub.g(t(5))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))

    with pytest.raises(ShapeMismatchError):
        sub.g(t(4))


def test_inherit_check_shapes__undefined() -> None:
    class SuperClass(ABC):
        pass

    with pytest.raises(RuntimeError):

        class SubClass(SuperClass):
            @inherit_check_shapes
            def f(self, a: Shaped) -> Shaped:
                return t(1)
