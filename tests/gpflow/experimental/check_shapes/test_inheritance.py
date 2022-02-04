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
from abc import ABC, abstractmethod

import pytest

from gpflow.base import TensorType
from gpflow.experimental.check_shapes import ShapeMismatchError, check_shapes, inherit_check_shapes

from .utils import t


def test_inherit_check_shapes__defined_in_super_class() -> None:
    class SuperClass(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: TensorType) -> TensorType:
            pass

    class MiddleClass(SuperClass):
        pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
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
        def f(self, a: TensorType) -> TensorType:
            pass

    class MiddleClass(SuperClass):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
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
        def f(self, a: TensorType) -> TensorType:
            pass

    class MiddleClass(SuperClass):
        def f(self, a: TensorType) -> TensorType:
            return t(2)

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
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
        def f(self, a: TensorType) -> TensorType:
            pass

    class SubClass(MiddleClass):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
            return t(1)

    sub = SubClass()
    sub.f(t(4))  # Don't crash...

    with pytest.raises(ShapeMismatchError):
        sub.f(t(5))


def test_inherit_check_shapes__multiple_inheritance() -> None:
    class Left(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [4]),
            ("return", [1]),
        )
        def f(self, a: TensorType) -> TensorType:
            pass

    class Right(ABC):
        @abstractmethod
        @check_shapes(
            ("a", [5]),
            ("return", [1]),
        )
        def g(self, a: TensorType) -> TensorType:
            pass

    class SubClass(Left, Right):
        @inherit_check_shapes
        def f(self, a: TensorType) -> TensorType:
            return t(1)

        @inherit_check_shapes
        def g(self, a: TensorType) -> TensorType:
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
            def f(self, a: TensorType) -> TensorType:
                return t(1)
