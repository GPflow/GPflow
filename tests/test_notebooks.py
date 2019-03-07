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

import glob
import os
import sys
import time
import traceback

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

from gpflow.test_util import session_context

# blacklisted notebooks should have a unique basename
BLACKLISTED_NOTEBOOKS = [
]


def _nbpath():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, '../doc/source/notebooks')


def get_notebooks():
    """
    Returns all notebooks in `_nbpath` that are not blacklisted.
    """
    def notebook_blacklisted(nb):
        blacklisted_notebooks_basename = map(os.path.basename, BLACKLISTED_NOTEBOOKS)
        return os.path.basename(nb) in blacklisted_notebooks_basename

    # recursively traverse the notebook directory in search for ipython notebooks
    all_notebooks = glob.iglob(os.path.join(_nbpath(), '**', '*.ipynb'), recursive=True)
    notebooks_to_test = [nb for nb in all_notebooks if not notebook_blacklisted(nb)]
    return notebooks_to_test


def _preproc():
    pythonkernel = 'python' + str(sys.version_info[0])
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel,
                               interrupt_on_timeout=True)


def _exec_notebook(notebook_filename):
    with open(notebook_filename) as notebook_file:
        nb = nbformat.read(notebook_file, as_version=nbformat.current_nbformat)
        try:
            meta_data = {'path': os.path.dirname(notebook_filename)}
            _preproc().preprocess(nb, {'metadata': meta_data})
        except CellExecutionError as cell_error:
            traceback.print_exc(file=sys.stdout)
            msg = 'Error executing the notebook {0}. See above for error.\nCell error: {1}'
            pytest.fail(msg.format(notebook_filename, str(cell_error)))


def _exec_notebook_ts(notebook_filename):
    with session_context():
        ts = time.time()
        _exec_notebook(notebook_filename)
        elapsed = time.time() - ts
        print(notebook_filename, 'took {0} seconds.'.format(elapsed))


@pytest.mark.notebooks
@pytest.mark.parametrize('notebook_file', get_notebooks())
def test_notebook(notebook_file):
    _exec_notebook_ts(notebook_file)

