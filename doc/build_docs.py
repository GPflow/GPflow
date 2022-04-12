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
Code for building GPflow's documentation for a specified branch.
"""
import argparse
import shutil
import subprocess
from itertools import chain
from pathlib import Path
from time import perf_counter
from typing import Optional

from generate_module_rst import generate_module_rst
from tabulate import tabulate
from versions import Branch

import gpflow

_SRC = Path(__file__).parent
_SPHINX_SRC = _SRC / "sphinx"
_NOTEBOOKS_SRC = _SPHINX_SRC / "notebooks"

_TMP = Path("/tmp/gpflow_build_docs")
_BUILD_TMP = _TMP / "build"
_NOTEBOOKS_TMP = _BUILD_TMP / "notebooks"
_DOCTREE_TMP = _TMP / "doctree"


def _create_fake_notebook(destination_relative_path: Path, max_notebooks: int) -> None:
    print(f"Generating fake, due to --max_notebooks={max_notebooks}")

    destination = _NOTEBOOKS_TMP / destination_relative_path
    title = f"Fake {destination.name}"
    title_line = "#" * len(title)

    destination.write_text(
        f"""{title}
{title_line}

Fake {destination.name} due to::

   --max_notebooks={max_notebooks}
"""
    )


def _build_notebooks(max_notebooks: Optional[int]) -> None:
    # Building the notebooks is really slow. Let's time it so we know which notebooks we can /
    # should optimise.
    timings = []
    all_notebooks = chain(_NOTEBOOKS_TMP.glob("**/*.pct.py"), _NOTEBOOKS_TMP.glob("**/*.md"))
    for i, source_path in enumerate(all_notebooks):
        before = perf_counter()
        print()
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        print("Building:", source_path)
        print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")

        source_relative_path = source_path.relative_to(_NOTEBOOKS_TMP)
        destination_relative_path = source_relative_path
        while destination_relative_path.suffix:  # .pct.py has several suffixes. Remove all of them.
            destination_relative_path = destination_relative_path.with_suffix("")
        destination_relative_path = destination_relative_path.with_suffix(".ipynb")

        if max_notebooks is None or i < max_notebooks:
            subprocess.run(
                [
                    "jupytext",
                    "--execute",
                    "--to",
                    "notebook",
                    "-o",
                    str(destination_relative_path),
                    str(source_relative_path),
                ],
                cwd=_NOTEBOOKS_TMP,
            ).check_returncode()
        else:
            _create_fake_notebook(destination_relative_path, max_notebooks)

        after = perf_counter()
        timings.append((after - before, source_relative_path))

    timings.sort(reverse=True)
    print()
    print("Notebooks by build-time:")
    print(tabulate(timings, headers=["Time", "Notebook"]))
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the GPflow documentation.")
    parser.add_argument(
        "branch",
        type=str,
        choices=[b.value for b in Branch],
        help="Git branch that is currently being built.",
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Directory to write docs to.",
    )
    parser.add_argument(
        "--max_notebooks",
        "--max-notebooks",
        type=int,
        help="Limit number of notebooks built to this number. Useful when debugging.",
    )
    parser.add_argument(
        "--fail_on_warning",
        "--fail-on-warning",
        default=False,
        action="store_true",
        help="If set, crash if there were any warnings while generating documentation.",
    )
    args = parser.parse_args()
    branch = Branch(args.branch)
    dest = args.destination
    version_dest = dest / branch.version

    shutil.rmtree(version_dest, ignore_errors=True)
    shutil.rmtree(_TMP, ignore_errors=True)
    _TMP.mkdir(parents=True)

    shutil.copytree(_SPHINX_SRC, _BUILD_TMP)
    (_BUILD_TMP / "build_version.txt").write_text(branch.version)
    _build_notebooks(args.max_notebooks)
    generate_module_rst(gpflow, _BUILD_TMP / "api")

    sphinx_commands = [
        "sphinx-build",
        "-b",
        "html",
        "-d",
        str(_DOCTREE_TMP),
        str(_BUILD_TMP),
        str(version_dest),
    ]
    if args.fail_on_warning:
        sphinx_commands.extend(
            [
                "-W",
                "--keep-going",
            ]
        )

    subprocess.run(sphinx_commands).check_returncode()


if __name__ == "__main__":
    main()
