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

from abc import ABC, abstractmethod
from contextlib import nullcontext  # pylint: disable=no-name-in-module
from typing import Any, Callable, ContextManager

import pytest

from gpflow.experimental.check_shapes import (
    check_shape,
    check_shapes,
    disable_check_shapes,
    inherit_check_shapes,
)
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError

from .utils import TestShaped, current_line, t


def test_inherit_check_shapes__defined_in_super_class() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class MiddleClass(SuperClass):  # pylint: disable=abstract-method
        pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__overridden_with_checks() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class MiddleClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__overridden_without_checks() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class MiddleClass(SuperClass):
        def f(self, a: TestShaped) -> TestShaped:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
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
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__multiple_inheritance() -> None:
    class Left(ABC):
        @abstractmethod
        @check_shapes(
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class Right(ABC):
        @abstractmethod
        @check_shapes(
            "a: [5]",
            "return: [1]",
        )
        def g(self, a: TestShaped) -> TestShaped:
            pass

    class SubClass(Left, Right):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            return t(1)

        @inherit_check_shapes
        def g(self, a: TestShaped) -> TestShaped:
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

        # pylint: disable=unused-variable

        class SubClass(SuperClass):
            @inherit_check_shapes
            def f(self, a: TestShaped) -> TestShaped:
                return t(1)


def test_inherit_check_shapes__check_shape() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [d]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    def_line = current_line() + 4
    call_line = current_line() + 5

    class SubClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            check_shape(t(3), "[d]")
            return t(1)

    sub = SubClass()
    sub.f(t(3))  # Don't crash...

    with pytest.raises(ShapeMismatchError) as e:
        sub.f(t(5))

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  Function:              test_inherit_check_shapes__check_shape.<locals>.SubClass.f
    Declared: {__file__}:{def_line}
    Argument: a
      Expected: [d]
      Actual:   [5]
  check_shape called at: {__file__}:{call_line}
    Expected: [d]
    Actual:   [3]
"""
        == message
    )


@pytest.mark.parametrize("enable_super", [True, False])
@pytest.mark.parametrize("enable_sub", [True, False])
def test_inherit_check_shapes__disable(enable_super: bool, enable_sub: bool) -> None:
    # Tests interactions between `disable_check_shapes` and `inherit_check_shapes`.

    typed_nullcontext: Callable[[], ContextManager[Any]] = nullcontext
    super_context: Callable[[], ContextManager[Any]] = (
        typed_nullcontext if enable_super else disable_check_shapes
    )
    sub_context: Callable[[], ContextManager[Any]] = (
        typed_nullcontext if enable_sub else disable_check_shapes
    )
    call_context: Callable[[], ContextManager[Any]] = (
        (lambda: pytest.raises(ShapeMismatchError))
        if (enable_super and enable_sub)
        else typed_nullcontext
    )

    with super_context():

        class SuperClass(ABC):
            @abstractmethod
            @check_shapes(
                "a: [d...]",
                "b: [d...]",
                "return: [d...]",
            )
            def f(self, a: TestShaped, b: TestShaped) -> TestShaped:
                pass

    with sub_context():

        class SubClass(SuperClass):
            @inherit_check_shapes
            def f(self, a: TestShaped, b: TestShaped) -> TestShaped:
                return a

    sub = SubClass()

    with call_context():
        sub.f(t(2, 3), t(2, 4))  # Wrong shape, checks maybe disabled.

    with disable_check_shapes():
        sub.f(t(2, 3), t(2, 4))  # Wrong shape, but checks disabled.


def test_inherit_check_shapes__error_message() -> None:
    # Here we're just testing that error message formatting is wired together sanely. For more
    # thorough tests of error formatting, see test_error_contexts.py and test_exceptions.py

    def_line = current_line() + 2

    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [d1, d2]",
            "b: [d1, d3]  # Some note on b",
            "return: [d2, d3]",
            "# Some note on f",
            "# Some other note on f",
        )
        def f(self, a: TestShaped, b: TestShaped) -> TestShaped:
            pass

    def_line = current_line() + 3

    class SubClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: TestShaped, b: TestShaped) -> TestShaped:
            return t(3, 4)

    sub = SubClass()

    with pytest.raises(ShapeMismatchError) as e:
        sub.f(t(2, 3), t(3, 4))

    (message,) = e.value.args
    assert (
        f"""
Tensor shape mismatch.
  Function: test_inherit_check_shapes__error_message.<locals>.SubClass.f
    Declared: {__file__}:{def_line}
    Note:     Some note on f
    Note:     Some other note on f
    Argument: a
      Expected: [d1, d2]
      Actual:   [2, 3]
    Argument: b
      Note:     Some note on b
      Expected: [d1, d3]
      Actual:   [3, 4]
"""
        == message
    )


def test_inherit_check_shapes__rewrites_docstring() -> None:
    # Here we're just testing that the rewrite is applied. For tests of formatting, see
    # test_parser.py

    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            "a: [4]",
            "return: [1]",
        )
        def f(self, a: TestShaped) -> TestShaped:
            pass

    class SubClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: TestShaped) -> TestShaped:
            """
            An inherited method.

            :param a: A parameter.
            :returns: A result.
            """
            return t(1)

    assert (
        """
            An inherited method.

            :param a:
                * **a** has shape [4].

                A parameter.
            :returns:
                * **return** has shape [1].

                A result.
            """
        == SubClass.f.__doc__
    )
