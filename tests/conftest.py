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
from typing import Iterable

import pytest
from _pytest.logging import LogCaptureFixture


@pytest.fixture(autouse=True)
def test_auto_graph_compile(caplog: LogCaptureFixture) -> Iterable[None]:
    yield

    for when in ["setup", "call", "teardown"]:
        for record in caplog.get_records(when):
            assert not record.msg.startswith("AutoGraph could not transform"), record.getMessage()
