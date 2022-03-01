# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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
from functools import wraps
from typing import Any, Callable, TypeVar, cast
from warnings import warn

C = TypeVar("C", bound=Callable[..., Any])


def experimental(func: C) -> C:
    """
    Decorator that marks the decorated function as experimental.

    The first time an experimental function is called, a warning is printed.

    Example::

        @experimental
        def forty_two() -> int:
            return 42
    """

    has_warned = False

    @wraps(func)
    def wrap_experimental(*args: Any, **kwargs: Any) -> Any:
        nonlocal has_warned

        if not has_warned:
            name = f"{func.__module__}.{func.__qualname__}"
            warn(
                f"You're calling {name} which is considered *experimental*."
                " Expect: breaking changes, poor documentation, and bugs."
            )
            has_warned = True

        return func(*args, **kwargs)

    return cast(C, wrap_experimental)
