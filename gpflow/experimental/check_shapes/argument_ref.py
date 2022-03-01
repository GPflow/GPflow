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
from typing import Any, Mapping

from .base_types import C
from .errors import ArgumentReferenceError

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

    def get(self, func: C, arg_map: Mapping[str, Any]) -> Any:
        """ Get the value of this argument from this given map. """
        try:
            return self._get(arg_map)
        except Exception as e:
            raise ArgumentReferenceError(func, arg_map, self) from e

    @abstractmethod
    def _get(self, arg_map: Mapping[str, Any]) -> Any:
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

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        return arg_map[self.argument_name]

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

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        # pylint: disable=protected-access
        return getattr(self.source._get(arg_map), self.attribute_name)

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

    def _get(self, arg_map: Mapping[str, Any]) -> Any:
        # pylint: disable=protected-access
        return self.source._get(arg_map)[self.index]

    def __repr__(self) -> str:
        return f"{repr(self.source)}[{self.index}]"
