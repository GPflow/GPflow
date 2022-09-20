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
from typing import AbstractSet

import pytest

from benchmark.registry import Named, Registry, Tagged, TaggedRegistry
from benchmark.tag import Tag


def test_registry() -> None:
    class TestNamed(Named):
        def __init__(self, name: str) -> None:
            self.name = name

    o1 = TestNamed("o1")
    o2 = TestNamed("o2")

    reg: Registry[TestNamed] = Registry()

    assert () == reg.names()
    assert () == reg.all()

    assert o1 == reg.add(o1)
    assert o2 == reg.add(o2)

    assert ("o1", "o2") == reg.names()
    assert (o1, o2) == reg.all()
    assert o1 == reg.get("o1")
    assert o2 == reg.get("o2")

    o1_2 = TestNamed("o1")
    with pytest.raises(AssertionError):
        reg.add(o1_2)


def test_tagged_registry() -> None:
    class TestTag(Tag["TestTag"]):
        pass

    t1 = TestTag("t1")
    t2 = TestTag("t2")

    class TestTagged(Tagged[TestTag]):
        def __init__(self, name: str, tags: AbstractSet[TestTag]) -> None:
            self.name = name
            self.tags = tags

    o1 = TestTagged("o1", {t1})
    o2 = TestTagged("o2", {t1, t2})
    o3 = TestTagged("o3", {t2})

    reg: TaggedRegistry[TestTagged, TestTag] = TaggedRegistry()

    assert () == reg.names()
    assert () == reg.all()
    assert () == reg.where(t1)
    assert () == reg.where(t1 & ~t2)

    assert o1 == reg.add(o1)
    assert o2 == reg.add(o2)
    assert o3 == reg.add(o3)

    assert ("o1", "o2", "o3") == reg.names()
    assert (o1, o2, o3) == reg.all()
    assert (o1, o2) == reg.where(t1)
    assert (o1,) == reg.where(t1 & ~t2)
    assert o1 == reg.get("o1")
    assert o2 == reg.get("o2")
    assert o3 == reg.get("o3")

    o1_2 = TestTagged("o1", {t2})
    with pytest.raises(AssertionError):
        reg.add(o1_2)
