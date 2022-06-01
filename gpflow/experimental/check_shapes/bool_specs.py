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
Code for specifying and evaluating boolean expressions.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Tuple, cast

from .argument_ref import ArgumentRef
from .error_contexts import ErrorContext, ObjectValueContext, ParallelContext, StackContext


class ParsedBoolSpec(ABC):
    """ A boolean expression. """

    @abstractmethod
    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        """ Evaluate this boolean value. """


def _paren_repr(spec: ParsedBoolSpec) -> str:
    """
    Return the `repr` of `spec`, wrapping it in parenthesis if it is a non-trivial expression.
    """
    result = repr(spec)
    if isinstance(spec, ParsedArgumentRefBoolSpec) and spec.bool_test == BoolTest.BOOL:
        return result
    return f"({result})"


@dataclass(frozen=True)
class ParsedOrBoolSpec(ParsedBoolSpec):
    """ An "or" expression. """

    left: ParsedBoolSpec
    right: ParsedBoolSpec

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        left_value, left_context = self.left.get(arg_map, context)
        right_value, right_context = self.right.get(arg_map, context)
        return (left_value or right_value), ParallelContext((left_context, right_context))

    def __repr__(self) -> str:
        return f"{_paren_repr(self.left)} or {_paren_repr(self.right)}"


@dataclass(frozen=True)
class ParsedAndBoolSpec(ParsedBoolSpec):
    """ An "and" expression. """

    left: ParsedBoolSpec
    right: ParsedBoolSpec

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        left_value, left_context = self.left.get(arg_map, context)
        right_value, right_context = self.right.get(arg_map, context)
        return (left_value and right_value), ParallelContext((left_context, right_context))

    def __repr__(self) -> str:
        return f"{_paren_repr(self.left)} and {_paren_repr(self.right)}"


@dataclass(frozen=True)
class ParsedNotBoolSpec(ParsedBoolSpec):
    """ A "not" expression. """

    right: ParsedBoolSpec

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        right_value, right_context = self.right.get(arg_map, context)
        return (not right_value), right_context

    def __repr__(self) -> str:
        return f"not {_paren_repr(self.right)}"


class BoolTest(Enum):
    """
    Strategies for converting a value to ``bool``.
    """

    BOOL = (bool, lambda arg: arg)
    IS_NONE = ((lambda x: x is None), lambda arg: f"{arg} is None")
    IS_NOT_NONE = ((lambda x: x is not None), lambda arg: f"{arg} is not None")

    @property
    def to_bool(self) -> Callable[[Any], bool]:
        return cast(Callable[[Any], bool], self.value[0])

    @property
    def repr(self) -> Callable[[str], str]:
        return cast(Callable[[str], str], self.value[1])


@dataclass(frozen=True)
class ParsedArgumentRefBoolSpec(ParsedBoolSpec):
    """ A reference to an input argument. """

    argument_ref: ArgumentRef
    bool_test: BoolTest

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        ((arg_value, relative_context),) = self.argument_ref.get(arg_map, context)
        bool_value = self.bool_test.to_bool(arg_value)
        return bool_value, StackContext(relative_context, ObjectValueContext(arg_value))

    def __repr__(self) -> str:
        return self.bool_test.repr(repr(self.argument_ref))
