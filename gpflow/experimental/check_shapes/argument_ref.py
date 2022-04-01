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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from .error_contexts import (
    ArgumentContext,
    AttributeContext,
    ErrorContext,
    IndexContext,
    StackContext,
)
from .exceptions import ArgumentReferenceError

# The special name used to represent the returned value. `return` is a good choice because we know
# that no argument can be called `return` because `return` is a reserved keyword.
RESULT_TOKEN = "return"


class ArgumentRef(ABC):
    """ A reference to an argument. """

    @property
    @abstractmethod
    def is_result(self) -> bool:
        """ Whether this is a reference to the function result. """

    @property
    @abstractmethod
    def root_argument_name(self) -> str:
        """
        Name of the argument this reference eventually starts from.

        Returns `RESULT_TOKEN` if this in an argument to the function result.
        """

    @property
    @abstractmethod
    def error_context(self) -> ErrorContext:
        """
        Return an error context for the value of this argument.
        """

    @abstractmethod
    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Optional[Any]:
        """ Get the value of this argument from this given map. """

    @abstractmethod
    def __repr__(self) -> str:
        """ Return a string representation of this reference. """


@dataclass(frozen=True)
class RootArgumentRef(ArgumentRef):
    """ A reference to a single argument. """

    argument_name: str

    @property
    def is_result(self) -> bool:
        return self.argument_name == RESULT_TOKEN

    @property
    def root_argument_name(self) -> str:
        return self.argument_name

    @property
    def error_context(self) -> ErrorContext:
        return ArgumentContext(self.argument_name)

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Optional[Any]:
        try:
            return arg_map[self.argument_name]
        except Exception as e:
            raise ArgumentReferenceError(StackContext(context, self.error_context)) from e

    def __repr__(self) -> str:
        return self.argument_name


@dataclass(frozen=True)
class AttributeArgumentRef(ArgumentRef):
    """ A reference to an attribute on an argument. """

    source: ArgumentRef
    attribute_name: str

    @property
    def is_result(self) -> bool:
        return self.source.is_result

    @property
    def root_argument_name(self) -> str:
        return self.source.root_argument_name

    @property
    def error_context(self) -> ErrorContext:
        return StackContext(self.source.error_context, AttributeContext(self.attribute_name))

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Optional[Any]:
        source = self.source.get(arg_map, context)
        if source is None:
            return None

        try:
            return getattr(source, self.attribute_name)
        except Exception as e:
            raise ArgumentReferenceError(StackContext(context, self.error_context)) from e

    def __repr__(self) -> str:
        return f"{repr(self.source)}.{self.attribute_name}"


@dataclass(frozen=True)
class IndexArgumentRef(ArgumentRef):
    """ A reference to an element in a list. """

    source: ArgumentRef
    index: int

    @property
    def is_result(self) -> bool:
        return self.source.is_result

    @property
    def root_argument_name(self) -> str:
        return self.source.root_argument_name

    @property
    def error_context(self) -> ErrorContext:
        return StackContext(self.source.error_context, IndexContext(self.index))

    def get(self, arg_map: Mapping[str, Any], context: ErrorContext) -> Optional[Any]:
        source = self.source.get(arg_map, context)
        if source is None:
            return None

        try:
            return source[self.index]  # type: ignore
        except Exception as e:
            raise ArgumentReferenceError(StackContext(context, self.error_context)) from e

    def __repr__(self) -> str:
        return f"{repr(self.source)}[{self.index}]"
