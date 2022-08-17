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

from benchmark.tag import NO_REQ, Tag, TagReq


class TestTag(Tag["TestTag"]):
    pass


t1 = TestTag("t1")
t2 = TestTag("t2")
t3 = TestTag("t3")


@pytest.mark.parametrize(
    "req,tags,expected_satisfied,expected_repr",
    [
        # Test single operations:
        (NO_REQ, {}, True, "NO_REQ"),
        (t1, {t1, t2}, True, "t1"),
        (t3, {t1, t2}, False, "t3"),
        (t1 & t2, {t1, t2}, True, "t1 & t2"),
        (t1 & t3, {t1, t2}, False, "t1 & t3"),
        (t1 | t2, {t2}, True, "t1 | t2"),
        (t1 | t3, {t2}, False, "t1 | t3"),
        (~t1, {t1, t2}, False, "~t1"),
        (~t3, {t1, t2}, True, "~t3"),
        # Test composition:
        (t1 | (t2 & ~t3), {}, False, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t3}, False, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t2}, True, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t2, t3}, False, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t1}, True, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t1, t3}, True, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t1, t2}, True, "t1 | (t2 & (~t3))"),
        (t1 | (t2 & ~t3), {t1, t2, t3}, True, "t1 | (t2 & (~t3))"),
    ],
)
def test_tag_req(
    req: TagReq[TestTag], tags: AbstractSet[TestTag], expected_satisfied: bool, expected_repr: str
) -> None:
    assert expected_satisfied == req.satisfied(tags)
    assert expected_repr == repr(req)
