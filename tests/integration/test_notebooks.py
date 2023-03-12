# Copyright 2017 the GPflow authors.
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

import itertools
import sys
import traceback
from pathlib import Path
from typing import Sequence, Set

import jupytext
import nbformat
import pytest
from nbclient.exceptions import CellExecutionError
from nbconvert.preprocessors.execute import ExecutePreprocessor

_NOTEBOOK_DIR = (Path(__file__) / "../../../doc/sphinx/notebooks").resolve()


def test_notebook_dir_exists() -> None:
    assert _NOTEBOOK_DIR.is_dir()


# To ignore a notebook, add its full base name (including .pct.py or .md
# extension, but without any directory component). NOTE that if there are
# several notebooks in different directories with the same base name, they will
# all get ignored (change the ignoring check to something else in that
# case, if need be!)
IGNORED_NOTEBOOKS: Set[str] = set()


def get_notebooks() -> Sequence[Path]:
    """
    Returns all notebooks in `_nbpath` that are not ignored.
    """

    def notebook_ignored(nb: Path) -> bool:
        return nb.name in IGNORED_NOTEBOOKS

    # recursively traverse the notebook directory in search for ipython notebooks
    all_py_notebooks = _NOTEBOOK_DIR.glob("**/*.pct.py")
    all_md_notebooks = _NOTEBOOK_DIR.glob("**/*.md")
    all_notebooks = itertools.chain(all_md_notebooks, all_py_notebooks)
    notebooks_to_test = [nb for nb in all_notebooks if not notebook_ignored(nb)]
    return notebooks_to_test


def _preproc() -> ExecutePreprocessor:
    pythonkernel = "python" + str(sys.version_info[0])
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel, interrupt_on_timeout=True)


def _exec_notebook(notebook_path: Path) -> None:
    with notebook_path.open() as notebook_file:
        nb = jupytext.read(notebook_file, as_version=nbformat.current_nbformat)
        try:
            meta_data = {"path": str(notebook_path.parent)}
            _preproc().preprocess(nb, {"metadata": meta_data})
        except CellExecutionError as cell_error:
            traceback.print_exc(file=sys.stdout)
            msg = "Error executing the notebook {0}. See above for error.\nCell error: {1}"
            pytest.fail(msg.format(str(notebook_path), str(cell_error)))


@pytest.mark.notebooks
@pytest.mark.parametrize("notebook_path", get_notebooks(), ids=str)
def test_notebook(notebook_path: Path) -> None:
    _exec_notebook(notebook_path)


def test_has_notebooks() -> None:
    assert len(get_notebooks()) >= 30, "there are probably some notebooks that were not discovered"
