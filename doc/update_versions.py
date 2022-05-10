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
Tool for updating the `versions.json` and `index.html` in our documentation.
"""
import argparse
import json
from pathlib import Path
from typing import Any, List, Mapping, Optional

from packaging.version import Version
from versions import Branch


class _Versions:
    def __init__(self, versions: List[Mapping[str, Any]]) -> None:
        self._versions = versions
        self._sorted = False

    @staticmethod
    def _versions_json(dest: Path) -> Path:
        return dest / "versions.json"

    @staticmethod
    def read(docs_dir: Path) -> "_Versions":
        if _Versions._versions_json(docs_dir).exists():
            with _Versions._versions_json(docs_dir).open("rt") as versions_file:
                versions = json.load(versions_file)
        else:
            versions = {}
        return _Versions(versions)

    def write(self, docs_dir: Path) -> None:
        with _Versions._versions_json(docs_dir).open("wt") as versions_file:
            json.dump(self._versions, versions_file, indent=2)

    def add(self, version: str) -> None:
        self._versions = [v for v in self._versions if v["version"] != version]
        self._versions.append(
            {
                "version": version,
                "url": f"https://gpflow.github.io/GPflow/{version}/index.html",
            }
        )
        self._sorted = False

    def sort(self) -> None:
        develop: Optional[Mapping[str, Any]] = None
        for v in self._versions:
            if v["version"] == "develop":
                develop = v
        versions = [v for v in self._versions if v["version"] != "develop"]
        versions.sort(key=lambda v: Version(v["version"]), reverse=True)
        if develop is not None:
            # Insert `develop` in the second spot, after the latest release, but before older
            # releases:
            versions.insert(1, develop)
        self._versions = versions
        self._sorted = True

    @property
    def latest(self) -> str:
        assert self._sorted, "Must sort versions to find latest."
        latest = self._versions[0]["version"]
        assert isinstance(latest, str)  # Hint for mypy.
        return latest


def _create_root_redirect(versions: _Versions, dest: Path) -> None:

    (dest / "index.html").write_text(
        f"""<html xmlns="http://www.w3.org/1999/xhtml">
  <head>
    <title>GPflow documentation</title>
    <meta http-equiv="refresh" content="0;URL='{versions.latest}/index.html'" />
  </head>
  <body>
    <p>Redirecting to the most recent release.</p>
  </body>
</html>
"""
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GPflow documentation.")
    parser.add_argument(
        "branch",
        type=str,
        choices=[b.value for b in Branch],
        help="Git branch that is currently being built.",
    )
    parser.add_argument("docs_dir", type=Path, help="To read / write docs versions from.")
    args = parser.parse_args()
    branch = Branch(args.branch)
    docs_dir = args.docs_dir

    versions = _Versions.read(docs_dir)
    versions.add(branch.version)
    versions.sort()
    versions.write(docs_dir)

    _create_root_redirect(versions, docs_dir)


if __name__ == "__main__":
    main()
