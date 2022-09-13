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
Code for registering stuff by name.

We have *a lot* of stuff with names in this project...
"""
from typing import AbstractSet, Any, Dict, Generic, Sequence, TypeVar

from typing_extensions import Protocol

from benchmark.tag import Tag, TagReq


class Named(Protocol):
    """
    Something with a name.
    """

    name: str


N = TypeVar("N", bound=Named)


class Registry(Generic[N]):
    """
    A registry of stuff with names.
    """

    def __init__(self) -> None:
        self._members: Dict[str, N] = {}

    def add(self, member: N) -> N:
        """
        Add the given member to this registry.

        Returns the input, for easy inlining.
        """
        assert (
            member.name not in self._members
        ), f"{member} already registred under name {member.name}."
        self._members[member.name] = member
        return member

    def get(self, name: str) -> N:
        """
        Get a member by name.
        """
        return self._members[name]

    def names(self) -> Sequence[str]:
        """
        Get a list of all registered names.
        """
        return tuple(self._members)

    def all(self) -> Sequence[N]:
        """
        Get a list of all registered values.
        """
        return tuple(self._members.values())


TTag = TypeVar("TTag", bound=Tag[Any])


class Tagged(Named, Protocol[TTag]):
    """
    Something with a name and some tags.
    """

    tags: AbstractSet[TTag]


T = TypeVar("T", bound=Tagged[Any])


class TaggedRegistry(Registry[T], Generic[T, TTag]):
    """
    A registry of stuff with names and tags.
    """

    def where(self, req: TagReq[TTag]) -> Sequence[T]:
        """
        Get all memebers that fulfill the given tag requirements.
        """
        return tuple(m for m in self._members.values() if req.satisfied(m.tags))
