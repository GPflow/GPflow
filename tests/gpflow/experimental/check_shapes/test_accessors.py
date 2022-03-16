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
import pytest

from gpflow.experimental.check_shapes.accessors import (
    get_check_shapes,
    maybe_get_check_shapes,
    set_check_shapes,
)
from gpflow.experimental.check_shapes.base_types import C


def test_get_set_check_shapes() -> None:
    def check_shapes(func: C) -> C:
        return func

    def func() -> None:
        pass

    assert maybe_get_check_shapes(func) is None
    with pytest.raises(ValueError):
        get_check_shapes(func)

    set_check_shapes(func, check_shapes)

    assert maybe_get_check_shapes(func) is check_shapes
    assert get_check_shapes(func) is check_shapes
