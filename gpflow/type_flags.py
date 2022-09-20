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

from numpy import __version__ as np_version
from packaging.version import Version

NP_VERSION = Version(np_version)

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


def compute_mypy_flags() -> str:  # pragma: no cover
    from mypy.version import __version__ as mypy_version

    MYPY_VERSION = Version(mypy_version)

    flags = []

    def set_always(variable: str, value: bool) -> None:
        if value:
            flags.append("--always-true")
        else:
            flags.append("--always-false")
        flags.append(variable)

    set_always("NP_TYPE_CHECKING", True)
    set_always("GENERIC_NP_ARRAYS", NP_VERSION >= Version("1.21.0"))
    if MYPY_VERSION >= Version("0.940"):
        flags.extend(["--enable-error-code", "ignore-without-code"])

    return " ".join(flags)
