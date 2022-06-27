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
Code for configuring check_shapes."
"""
from contextlib import contextmanager
from enum import Enum
from typing import Iterator, Union

import tensorflow as tf


class ShapeCheckingState(Enum):
    """
    Different states of whether to actually check shapes.
    """

    ENABLED = "enabled"
    """
    Always check shapes.
    """

    EAGER_MODE_ONLY = "eager_mode_only"
    """
    Only check shapes if `tf.inside_function()` is `False`.
    """

    DISABLED = "disabled"
    """
    Never check shapes.
    """

    def __bool__(self) -> bool:
        """
        Return whether we currently should check shapes.
        """
        if self is ShapeCheckingState.ENABLED:
            return True
        elif self is ShapeCheckingState.EAGER_MODE_ONLY:
            # pylint: disable=no-member
            return not tf.inside_function()
        else:
            assert self is ShapeCheckingState.DISABLED, self
            return False


_enabled = ShapeCheckingState.EAGER_MODE_ONLY


class DocstringFormat(Enum):
    """
    Enumeration of supported formats of docstrings.
    """

    SPHINX = "sphinx"
    """
    Rewrite docstrings in the `Sphinx <https://www.sphinx-doc.org/en/master/>`_ format.
    """

    NONE = "none"
    """
    Do not rewrite docstrings.
    """


_docstring_format = DocstringFormat.SPHINX

_function_call_precompute_enabled = False


def set_enable_check_shapes(enabled: Union[ShapeCheckingState, str, bool]) -> None:
    """
    Set whether to enable :mod:`check_shapes`.

    Check shapes has a non-zero impact on performance. If this is unacceptable to you, you can
    use this function to disable it.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [disable__manual]
       :end-before: [disable__manual]
       :dedent:

    See also :func:`disable_check_shapes`.
    """
    global _enabled
    if enabled == True:  # pylint: disable=singleton-comparison
        _enabled = ShapeCheckingState.ENABLED
    elif enabled == False:  # pylint: disable=singleton-comparison
        _enabled = ShapeCheckingState.DISABLED
    elif isinstance(enabled, str):
        _enabled = ShapeCheckingState(enabled)
    else:
        assert isinstance(enabled, ShapeCheckingState), type(ShapeCheckingState)
        _enabled = enabled


def get_enable_check_shapes() -> ShapeCheckingState:
    """
    Get whether to enable :mod:`check_shapes`.
    """
    return _enabled


@contextmanager
def disable_check_shapes() -> Iterator[None]:
    """
    Context manager that temporarily disables shape checking.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [disable__context_manager]
       :end-before: [disable__context_manager]
       :dedent:
    """
    old_value = get_enable_check_shapes()
    set_enable_check_shapes(ShapeCheckingState.DISABLED)
    try:
        yield
    finally:
        set_enable_check_shapes(old_value)


def set_rewrite_docstrings(docstring_format: Union[DocstringFormat, str, None]) -> None:
    """
    Set how :mod:`check_shapes` should rewrite docstrings.

    See :class:`DocstringFormat` for valid choices.
    """
    global _docstring_format
    if docstring_format is None:
        _docstring_format = DocstringFormat.NONE
    elif isinstance(docstring_format, str):
        _docstring_format = DocstringFormat(docstring_format)
    else:
        assert isinstance(docstring_format, DocstringFormat), type(docstring_format)
        _docstring_format = docstring_format


def get_rewrite_docstrings() -> DocstringFormat:
    """
    Get how :mod:`check_shapes` should rewrite docstrings.
    """
    return _docstring_format


def set_enable_function_call_precompute(enabled: bool) -> None:
    """
    Set whether to precompute function call path and line numbers for debugging.

    This is disabled by default, because it is (relatively) slow. Enabling this can give better
    error messages.

    Example:

    .. literalinclude:: /examples/test_check_shapes_examples.py
       :start-after: [disable_function_call_precompute]
       :end-before: [disable_function_call_precompute]
       :dedent:
    """
    global _function_call_precompute_enabled
    _function_call_precompute_enabled = enabled


def get_enable_function_call_precompute() -> bool:
    """
    Get whether to precompute function call path and line numbers for debugging.
    """
    return _function_call_precompute_enabled
