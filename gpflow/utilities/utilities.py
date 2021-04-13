# Copyright 2017-2021 The GPflow Contributors. All Rights Reserved.
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
This module is deprecated, and is only provided for backwards compatibility.
It will be removed in GPflow 2.3.
"""
from deprecated import deprecated

from . import misc, traversal

__all__ = []


def _create_module_redirects(m):
    for name in m.__all__:
        func = getattr(m, name)
        assert callable(func), "all names exported by misc and traversal should be functions"
        deprecated_func = deprecated(
            reason="The gpflow.utilities.utilities module is deprecated and will "
            f"be removed in GPflow 2.3; use gpflow.utilities.{name} instead."
        )(func)
        globals()[name] = deprecated_func
        __all__.append(name)


_create_module_redirects(misc)
_create_module_redirects(traversal)
del _create_module_redirects, misc, traversal
