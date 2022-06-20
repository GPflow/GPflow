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
from typing import Any, Iterable, List, Mapping, Sequence, Tuple

from .error_contexts import (
    ArgumentContext,
    AttributeContext,
    ErrorContext,
    IndexContext,
    MappingKeyContext,
    MappingValueContext,
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

    @abstractmethod
    def get(
        self, arg_map: Mapping[str, Any], context: ErrorContext
    ) -> Sequence[Tuple[Any, ErrorContext]]:
        """
        Get the value(s) of this argument from the given argument map.
        """

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

    def get(
        self, arg_map: Mapping[str, Any], context: ErrorContext
    ) -> Sequence[Tuple[Any, ErrorContext]]:
        relative_context = ArgumentContext(self.argument_name)

        try:
            arg_value = arg_map[self.argument_name]
        except Exception as e:
            raise ArgumentReferenceError(StackContext(context, relative_context)) from e

        return [(arg_value, relative_context)]

    def __repr__(self) -> str:
        return self.argument_name


@dataclass(frozen=True)  # type: ignore[misc]
class DelegatingArgumentRef(ArgumentRef):
    """ Abstract base class for :class:`ArgumentRef`\ s that delegates to a source. """

    source: ArgumentRef

    @property
    def is_result(self) -> bool:
        return self.source.is_result

    @property
    def root_argument_name(self) -> str:
        return self.source.root_argument_name

    @abstractmethod
    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        """
        Map this value, from `self.source` to new value(s).
        """

    def map_context(self, context: ErrorContext) -> ErrorContext:
        """
        Pre-map this error context from `self.source`.

        The mapped value will both be used for error messages and passed to `map_value` above.
        """
        return context

    def get(
        self, arg_map: Mapping[str, Any], context: ErrorContext
    ) -> Sequence[Tuple[Any, ErrorContext]]:
        results: List[Tuple[Any, ErrorContext]] = []
        sources = self.source.get(arg_map, context)
        for source, source_relative_context in sources:

            if source is None:
                results.append((source, source_relative_context))
                continue

            try:
                relative_context = self.map_context(source_relative_context)
            except Exception as e:
                raise ArgumentReferenceError(context) from e

            try:
                results.extend(self.map_value(source, relative_context))
            except Exception as e:
                raise ArgumentReferenceError(StackContext(context, relative_context)) from e

        return results


@dataclass(frozen=True)
class AttributeArgumentRef(DelegatingArgumentRef):
    """ A reference to an attribute on an argument. """

    attribute_name: str

    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        return [(getattr(value, self.attribute_name), context)]

    def map_context(self, context: ErrorContext) -> ErrorContext:
        return StackContext(context, AttributeContext(self.attribute_name))

    def __repr__(self) -> str:
        return f"{self.source!r}.{self.attribute_name}"


@dataclass(frozen=True)
class IndexArgumentRef(DelegatingArgumentRef):
    """ A reference to an element in a list. """

    index: int

    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        return [(value[self.index], context)]

    def map_context(self, context: ErrorContext) -> ErrorContext:
        return StackContext(context, IndexContext(self.index))

    def __repr__(self) -> str:
        return f"{self.source!r}[{self.index}]"


@dataclass(frozen=True)
class AllElementsRef(DelegatingArgumentRef):
    """ A reference to all elements in a collection. """

    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        return [(v, StackContext(context, IndexContext(i))) for i, v in enumerate(value)]

    def __repr__(self) -> str:
        return f"{self.source!r}[all]"


@dataclass(frozen=True)
class KeysRef(DelegatingArgumentRef):
    """ A reference to all keys of a mapping. """

    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        return [(k, StackContext(context, MappingKeyContext(k))) for k in value]

    def __repr__(self) -> str:
        return f"{self.source!r}.keys()"


@dataclass(frozen=True)
class ValuesRef(DelegatingArgumentRef):
    """ A reference to all values of a mapping. """

    def map_value(self, value: Any, context: ErrorContext) -> Iterable[Tuple[Any, ErrorContext]]:
        return [(v, StackContext(context, MappingValueContext(k))) for k, v in value.items()]

    def __repr__(self) -> str:
        return f"{self.source!r}.values()"
