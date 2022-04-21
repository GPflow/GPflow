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
Code for setting type hints, depending on library versions.
"""
import sys

import numpy as np
from packaging.version import Version

NP_VERSION = Version(np.__version__)  # type: ignore

NP_TYPE_CHECKING = False
"""
Whether to type-check numpy arrays at all. Defaults to False, because we don't know which
versions a client might use.
"""

GENERIC_NP_ARRAYS = (sys.version_info >= (3, 9)) and (NP_VERSION >= Version("1.22.0"))
"""
Whether to use generic numpy arrays. This is not applied at all, if type checking and not
`NP_TYPE_CHECKING`.
"""

MYPY_FLAGS = {
    "NP_TYPE_CHECKING": True,
    "GENERIC_NP_ARRAYS": NP_VERSION >= Version("1.21.0"),
}
