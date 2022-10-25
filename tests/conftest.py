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
from check_shapes.config import (
    DocstringFormat,
    ShapeCheckingState,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)


@pytest.fixture(autouse=True)
def test_auto_graph_compile(caplog: LogCaptureFixture) -> Iterable[None]:
    yield

    for when in ["setup", "call", "teardown"]:
        # Type ignore below is because `when` should have type
        # `Literal['setup', 'call', 'teardown']`. We should be able to remove this when we no longer
        # need to support Python 3.7.
        for record in caplog.get_records(when):  # type: ignore[arg-type]
            assert not record.msg.startswith("AutoGraph could not transform"), record.getMessage()


@pytest.fixture(autouse=True)
def enable_shape_checks() -> Iterable[None]:
    # Ensure that:
    # 1: `check_shapes` is enabled when running these tests.
    # 2: If a test manipulates `check_shapes` settings, they are reset after the test.
    # See also: tests/conftest.py
    old_enable = get_enable_check_shapes()
    old_rewrite_docstrings = get_rewrite_docstrings()
    old_function_call_precompute = get_enable_function_call_precompute()
    set_enable_check_shapes(ShapeCheckingState.ENABLED)
    set_rewrite_docstrings(DocstringFormat.SPHINX)
    set_enable_function_call_precompute(True)
    yield
    set_enable_function_call_precompute(old_function_call_precompute)
    set_rewrite_docstrings(old_rewrite_docstrings)
    set_enable_check_shapes(old_enable)
