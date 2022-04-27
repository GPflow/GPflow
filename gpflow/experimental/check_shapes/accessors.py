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
Utilities for setting and getting `check_shapes`
"""
from typing import Any, Callable, Optional

from .base_types import C


def set_check_shapes(func: Callable[..., Any], check_shapes: Callable[[C], C]) -> None:
    """
    Store ``check_shapes`` in ``func``, so that we later can tell which ``check_shapes`` was applied
    to it.
    """
    setattr(func, "__check_shapes__", check_shapes)


def maybe_get_check_shapes(func: Callable[..., Any]) -> Optional[Callable[[C], C]]:
    """
    Get the ``check_shapes`` that was applied to ``func``.

    :returns: The ``check_shapes`` that is wrapping ``func``, and ``None`` if no ``check_shapes`` is
        not wrapping ``func``.
    """
    return getattr(func, "__check_shapes__", None)


def get_check_shapes(func: Callable[..., Any]) -> Callable[[C], C]:
    """
    Get the ``check_shapes`` that was applied to ``func``.

    :raises ValueError: If no ``check_shapes`` was applied to ``func``.
    """
    result = maybe_get_check_shapes(func)
    if result is None:
        raise ValueError(f"{func.__name__} does not have a `check_shapes`.")
    return result
