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

from gpflow.experimental.check_shapes.config import (
    DocstringFormat,
    disable_check_shapes,
    get_enable_check_shapes,
    get_rewrite_docstrings,
    set_enable_check_shapes,
    set_rewrite_docstrings,
)


def test_get_set_enable_check_shapes() -> None:
    assert get_enable_check_shapes()
    set_enable_check_shapes(False)
    assert not get_enable_check_shapes()
    set_enable_check_shapes(True)
    assert get_enable_check_shapes()


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


def test_get_set_rewrite_docstrings() -> None:
    assert DocstringFormat.SPHINX == get_rewrite_docstrings()
    set_rewrite_docstrings(None)
    assert DocstringFormat.NONE == get_rewrite_docstrings()
    set_rewrite_docstrings("sphinx")
    assert DocstringFormat.SPHINX == get_rewrite_docstrings()
    set_rewrite_docstrings("none")
    assert DocstringFormat.NONE == get_rewrite_docstrings()
    set_rewrite_docstrings(DocstringFormat.SPHINX)
    assert DocstringFormat.SPHINX == get_rewrite_docstrings()
    set_rewrite_docstrings(DocstringFormat.NONE)
    assert DocstringFormat.NONE == get_rewrite_docstrings()
