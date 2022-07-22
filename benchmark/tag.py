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
Infrastructure for tagging stuff, and checking whether a set of tags fulfill some criteria.
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Generic, Iterable, List, TypeVar

T = TypeVar("T")


class TagReq(Generic[T], ABC):
    """
    Requirements on tags.

    A ``TagReq`` may or may not be "satisfied" by a set of tags.

    A ``Tag`` is itself a ``TagReq`` that is satisfied by itself::

        t1.satisfied({t1})  # True
        t1.satisfied({t2})  # False

    A ``TagReq`` can be inverted by the ``~`` operator::

        (~t1).satisfied({t2})  # True
        (~t1).satisfied({t1})  # False

    You can take the disjunction of two ``TagReq`` with the ``&`` operator::

        (t1 & t2).satisfied({t1, t2})  # True
        (t1 & t2).satisfied({t1})  # False

    You can take the union of two ``TagReq`` with the ``|`` operator::

        (t1 | t2).satisfied({t1})  # True
        (t1 | t2).satisfied({t3})  # False

    Finally there's a special ``TagReq`` called ``NO_REQ`` that always is satisfied:

        NO_REQ.satisfied({})  # True

    """

    @abstractmethod
    def satisfied(self, tags: Collection["Tag[T]"]) -> bool:
        """ Return whether this requirement is satisfied by the given ``tags``. """

    def child_repr(self) -> str:
        """
        Do a pretty ``repr`` of a wrapped child requirement.

        Basically wraps the child in ``()`` if it is "complicated".
        """
        if isinstance(self, (Tag, NoTagReq)):
            return repr(self)
        return "(" + repr(self) + ")"

    def __and__(self, other: "TagReq[T]") -> "TagReq[T]":
        return ReducingTagReq.create_and(self, other)

    def __or__(self, other: "TagReq[T]") -> "TagReq[T]":
        return ReducingTagReq.create_or(self, other)

    def __invert__(self) -> "TagReq[T]":
        return NotTagReq(self)


class Tag(TagReq[T], Generic[T]):
    """
    A tag that can be included in requirements.

    Usually you want to subclass this for type safety::

        class MyTag(Tag["MyTag"]):
            pass

        my_tag_1 = MyTag("my_tag_1")
        my_tag_2 = MyTag("my_tag_2")
        ...

    See :class:``TagReq`` for how to define requirements on tags.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def satisfied(self, tags: Collection["Tag[T]"]) -> bool:
        return self in tags

    def __repr__(self) -> str:
        return self.name


class ReducingTagReq(TagReq[T], Generic[T]):
    """
    ``TagReq`` that combines several children.
    """

    def __init__(
        self, name: str, reducer: Callable[[Iterable[bool]], bool], children: Collection[TagReq[T]]
    ) -> None:
        self.name = name
        self.reducer = reducer
        self.children: List[TagReq[T]] = []
        for child in children:
            if isinstance(child, ReducingTagReq) and self.reducer == child.reducer:
                self.children.extend(child.children)
            else:
                self.children.append(child)

    @staticmethod
    def create_and(*children: TagReq[T]) -> TagReq[T]:
        return ReducingTagReq("&", all, children)

    @staticmethod
    def create_or(*children: TagReq[T]) -> TagReq[T]:
        return ReducingTagReq("|", any, children)

    def satisfied(self, tags: Collection[Tag[T]]) -> bool:
        return self.reducer(c.satisfied(tags) for c in self.children)

    def __repr__(self) -> str:
        return f" {self.name} ".join(f"{c.child_repr()}" for c in self.children)


class NotTagReq(TagReq[T], Generic[T]):
    """
    ``TagReq`` that inverts a child.
    """

    def __init__(self, child: TagReq[T]) -> None:
        self._child = child

    def satisfied(self, tags: Collection[Tag[T]]) -> bool:
        return not self._child.satisfied(tags)

    def __repr__(self) -> str:
        return f"~{self._child.child_repr()}"


class NoTagReq(TagReq[Any]):
    """
    ``TagReq`` that always is satisfied.
    """

    def satisfied(self, tags: Collection[Tag[T]]) -> bool:
        return True

    def __repr__(self) -> str:
        return "NO_REQ"


NO_REQ = NoTagReq()
""" Singleton no-op ``TagReq`` that always is satisfied. """
