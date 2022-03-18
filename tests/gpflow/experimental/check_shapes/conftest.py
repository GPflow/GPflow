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

from gpflow.experimental.check_shapes.config import (
    DocstringFormat,
    get_enable_check_shapes,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_rewrite_docstrings,
)


@pytest.fixture(autouse=True)
def reset_settings() -> Iterable[None]:
    # Ensure that:
    # 1: `check_shapes` is enabled when running these tests.
    # 2: If a test manipulates `check_shapes` settings, they are reset after the test.
    old_enable = get_enable_check_shapes()
    old_rewrite_docstrings = get_rewrite_docstrings()
    set_enable_check_shapes(True)
    set_rewrite_docstrings(DocstringFormat.SPHINX)
    yield
    set_rewrite_docstrings(old_rewrite_docstrings)
    set_enable_check_shapes(old_enable)
