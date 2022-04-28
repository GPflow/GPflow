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
Code for keeping track of different versions of documentation.
"""
from enum import Enum
from pathlib import Path

# gpflow.__version__ does not seem reliable when used from a branch, so we read the VERSION file
# directly. (That's also a lot faster than importing gpflow...)
_GPFLOW_VERSION_FILE = Path(__file__).parent.parent / "VERSION"


class Branch(Enum):
    DEVELOP = "develop"
    """
    Building the `develop` branch means we're creating temporary documentation.

    Use this if in doubt.
    """

    MASTER = "master"
    """
    Building the `master` branch means we're creating a new release.
    """

    @property
    def version(self) -> str:
        """
        Returns the version of code / documentation this branch refers to.
        """
        if self == Branch.DEVELOP:
            return "develop"
        else:
            assert self == Branch.MASTER
            return _GPFLOW_VERSION_FILE.read_text().strip()
