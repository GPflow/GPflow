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
from typing import Any, Mapping, Tuple

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
    if isinstance(spec, ParsedArgumentRefBoolSpec):
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


@dataclass(frozen=True)
class ParsedArgumentRefBoolSpec(ParsedBoolSpec):
    """ A reference to an input argument. """

    argument_ref: ArgumentRef

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Tuple[bool, ErrorContext]:
        ((arg_value, relative_context),) = self.argument_ref.get(arg_map, context)
        return bool(arg_value), StackContext(relative_context, ObjectValueContext(arg_value))

    def __repr__(self) -> str:
        return repr(self.argument_ref)
