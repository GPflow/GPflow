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
Code for inheriting shape checks from a super class.
"""
import inspect
from typing import Callable, Optional, cast

from ..utils import experimental
from .base_types import C


@experimental
def inherit_check_shapes(func: C) -> C:
    """
    Decorator that inherits the :func:`check_shapes` decoration from any overridden method in a
    super-class.

    See: `Class inheritance`_.
    """
    return cast(C, _InheritCheckShapes(func))


class _InheritCheckShapes:
    """
    Implementation of inherit_check_shapes.

    The ``__set_name__`` hack is to get access to the class the method was declared on.
    See: https://stackoverflow.com/a/54316392 .
    """

    def __init__(self, func: C) -> None:
        self._func = func

    def __set_name__(self, owner: type, name: str) -> None:
        overridden_check_shapes: Optional[Callable[[C], C]] = None
        for parent in inspect.getmro(owner)[1:]:
            overridden_method = getattr(parent, name, None)
            if overridden_method is None:
                continue
            overridden_check_shapes = getattr(overridden_method, "__check_shapes__", None)
            if overridden_check_shapes is None:
                continue
            break

        assert overridden_check_shapes is not None, (
            f"@inherit_check_shapes did not find any overridden method of name '{name}'"
            f" on class '{owner.__name__}'."
        )

        self._func.class_name = owner.__name__  # type: ignore
        wrapped = overridden_check_shapes(self._func)
        setattr(owner, name, wrapped)
