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
Code for (de)referencing arguments.
"""
import re
from abc import ABC, abstractmethod
from typing import Any, Mapping, Match, Optional, Pattern

from .base_types import C
from .errors import ArgumentReferenceError

# The special name used to represent the returned value. `return` is a good choice because we know
# that no argument can be called `return` because `return` a reserved keyword.
RESULT_TOKEN = "return"

_NAME_RE_STR = "([_a-zA-Z][_a-zA-Z0-9]*)"
_ROOT_ARGUMENT_RE = re.compile(_NAME_RE_STR)
_ATTRIBUTE_ARGUMENT_RE = re.compile(f"\\.{_NAME_RE_STR}")
_INDEX_ARGUMENT_RE = re.compile(r"\[(\d+)\]")


class ArgumentRef(ABC):
    """ A reference to an argument. """

    @property
    @abstractmethod
    def is_result(self) -> bool:
        """ Whether this is a reference to the function result. """

    def get(self, func: C, arg_map: Mapping[str, Any]) -> Any:
        """ Get the value of this argument from this given map. """
        try:
            return self._get(arg_map)
        except Exception as e:
            raise ArgumentReferenceError(func, arg_map, self) from e

    @abstractmethod
    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        """ Get the value of this argument from this given map. """


class RootArgumentRef(ArgumentRef):
    """ A reference to a single argument. """

    def __init__(self, argument_name: str) -> None:
        self._argument_name = argument_name

    @property
    def is_result(self) -> bool:
        return self._argument_name == RESULT_TOKEN

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return arg_map[self._argument_name]

    def __repr__(self) -> str:
        return self._argument_name


class AttributeArgumentRef(ArgumentRef):
    """ A reference to an attribute on an argument. """

    def __init__(self, source: ArgumentRef, attribute_name: str) -> None:
        self._source = source
        self._attribute_name = attribute_name

    @property
    def is_result(self) -> bool:
        return self._source.is_result

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return getattr(self._source._get(arg_map), self._attribute_name)

    def __repr__(self) -> str:
        return f"{repr(self._source)}.{self._attribute_name}"


class IndexArgumentRef(ArgumentRef):
    """ A reference to an element in a list. """

    def __init__(self, source: ArgumentRef, index: int) -> None:
        self._source = source
        self._index = index

    @property
    def is_result(self) -> bool:
        return self._source.is_result

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return self._source._get(arg_map)[self._index]

    def __repr__(self) -> str:
        return f"{repr(self._source)}[{self._index}]"


def parse_argument_ref(argument_ref_str: str) -> ArgumentRef:
    def _create_error_message() -> str:
        return f"Invalid argument reference: '{argument_ref_str}'."

    start = 0
    match: Optional[Match[str]] = None

    def _consume(expression: Pattern[str]) -> None:
        nonlocal start, match
        match = expression.match(argument_ref_str, start)
        if match:
            start = match.end()

    _consume(_ROOT_ARGUMENT_RE)
    assert match, _create_error_message()
    result: ArgumentRef = RootArgumentRef(match.group(0))
    while start < len(argument_ref_str):
        _consume(_ATTRIBUTE_ARGUMENT_RE)
        if match:
            result = AttributeArgumentRef(result, match.group(1))
            continue
        _consume(_INDEX_ARGUMENT_RE)
        if match:
            result = IndexArgumentRef(result, int(match.group(1)))
            continue
        assert False, _create_error_message()
    assert start == len(argument_ref_str), _create_error_message()
    return result
