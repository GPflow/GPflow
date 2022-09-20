# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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

# pylint: skip-file

import os
from typing import Any, Iterable, List, Sequence, Type, TypeVar


def is_continuous_integration() -> bool:
    """
    Determines whether we are running on the Continuous Integration system for
    notebook integration tests. This is used to speed up notebook integration
    tests (built on every pull request commit) by capping all expensive loops
    at a small number, rather than running until convergence. When building the
    docs (indicated by the presence of the `DOCS` environment variable), we
    need to run notebooks to completion, and this function returns `False`.
    Whether we are running on CI is determined by the presence of the `CI`
    environment variable.
    """
    if "DOCS" in os.environ:
        return False

    return "CI" in os.environ


def reduce_in_tests(n: int, test_n: int = 2) -> int:
    return test_n if is_continuous_integration() else n


def subclasses(cls: Type[Any]) -> Iterable[Type[Any]]:
    """
    Generator that returns all (not just direct) subclasses of `cls`
    """
    for subclass in cls.__subclasses__():
        yield from subclasses(subclass)
        yield subclass
