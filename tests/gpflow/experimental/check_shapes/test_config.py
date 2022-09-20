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
from typing import Union

import pytest
import tensorflow as tf

from gpflow.experimental.check_shapes.config import (
    DocstringFormat,
    ShapeCheckingState,
    disable_check_shapes,
    get_enable_check_shapes,
    get_enable_function_call_precompute,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_enable_function_call_precompute,
    set_rewrite_docstrings,
)


@pytest.mark.parametrize(
    "state,eager_expected,function_expected",
    [
        (ShapeCheckingState.ENABLED, True, True),
        (ShapeCheckingState.EAGER_MODE_ONLY, True, False),
        (ShapeCheckingState.DISABLED, False, False),
    ],
)
def test_shape_checking_state__bool(
    state: ShapeCheckingState, eager_expected: bool, function_expected: bool
) -> None:
    enabled = None

    def run() -> None:
        nonlocal enabled
        enabled = bool(state)

    run()
    assert eager_expected == enabled

    tf.function(run)()  # pylint: disable=no-member
    assert function_expected == enabled


@pytest.mark.parametrize(
    "enable,expected",
    [
        (True, ShapeCheckingState.ENABLED),
        (False, ShapeCheckingState.DISABLED),
        ("enabled", ShapeCheckingState.ENABLED),
        ("eager_mode_only", ShapeCheckingState.EAGER_MODE_ONLY),
        ("disabled", ShapeCheckingState.DISABLED),
        (ShapeCheckingState.ENABLED, ShapeCheckingState.ENABLED),
        (ShapeCheckingState.EAGER_MODE_ONLY, ShapeCheckingState.EAGER_MODE_ONLY),
        (ShapeCheckingState.DISABLED, ShapeCheckingState.DISABLED),
    ],
)
def test_get_set_enable_check_shapes(
    enable: Union[ShapeCheckingState, str, bool], expected: ShapeCheckingState
) -> None:
    set_enable_check_shapes(enable)
    assert expected == get_enable_check_shapes()


def test_disable_check_shapes() -> None:
    assert get_enable_check_shapes()

    with disable_check_shapes():
        assert not get_enable_check_shapes()
        with disable_check_shapes():
            assert not get_enable_check_shapes()
        assert not get_enable_check_shapes()

    assert get_enable_check_shapes()

    with pytest.raises(ValueError):
        with disable_check_shapes():
            assert not get_enable_check_shapes()
            raise ValueError("test error")

    assert get_enable_check_shapes()


@pytest.mark.parametrize(
    "docstring_format,expected",
    [
        (None, DocstringFormat.NONE),
        ("sphinx", DocstringFormat.SPHINX),
        ("none", DocstringFormat.NONE),
        (DocstringFormat.NONE, DocstringFormat.NONE),
        (DocstringFormat.SPHINX, DocstringFormat.SPHINX),
    ],
)
def test_get_set_rewrite_docstrings(
    docstring_format: Union[DocstringFormat, str, None], expected: DocstringFormat
) -> None:
    set_rewrite_docstrings(docstring_format)
    assert expected == get_rewrite_docstrings()


@pytest.mark.parametrize(
    "enabled,expected",
    [
        (True, True),
        (False, False),
    ],
)
def test_get_set_enable_function_call_precompute(enabled: bool, expected: bool) -> None:
    set_enable_function_call_precompute(enabled)
    assert expected == get_enable_function_call_precompute()
