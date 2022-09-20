# Copyright 2022 the GPflow authors.
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
import subprocess
from pathlib import Path

import pytest

BUILD_DOCS_PATH = Path(__file__).parent.parent.parent / "doc" / "build_docs.py"


@pytest.mark.docs
def test_docs(tmp_path: Path) -> None:
    assert BUILD_DOCS_PATH.is_file()
    build_path = tmp_path / "build"

    try:
        subprocess.run(
            [
                "python",
                str(BUILD_DOCS_PATH),
                "develop",
                str(build_path),
                "--limit-notebooks",
                "--fail-on-warning",
            ]
        ).check_returncode()
    except subprocess.CalledProcessError as e:
        raise AssertionError(
            "Documentation build had errors / warnings."
            " Please fix."
            " Check both any .rst files you may have modified directly,"
            " and docstrings of your Python code."
        ) from e
