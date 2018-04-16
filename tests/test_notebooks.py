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
# limitations under the License.from __future__ import print_function

import traceback
import sys
import time
import os

import pytest
import nbformat

from gpflow.test_util import session_context

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

NOTEBOOK_FILES = [
    "classification.ipynb",
    "coreg_demo.ipynb",
    "kernels.ipynb",
    "mcmc.ipynb",
    "ordinal.ipynb",
    "Sanity_check.ipynb",
    "settings.ipynb",
    "SGPR_notes.ipynb",
    "vgp_notes.ipynb",
    "models.ipynb",
    "multiclass.ipynb",
    "classification.ipynb"
    # Blacklist:
    #   "FITCvsVFE.ipynb",
    #   "svi.ipynb",
    #   "GPLVM.ipynb",
    #   "regression.ipynb"
]


@pytest.mark.parametrize('notebook_file', NOTEBOOK_FILES)
def test_notebook(notebook_file):
    _exec_notebook_ts(notebook_file)


def _nbpath():
    this_dir = os.path.dirname(__file__)
    return os.path.join(this_dir, '../doc/source/notebooks/')


def _preproc():
    pythonkernel = 'python' + str(sys.version_info[0])
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel,
                               interrupt_on_timeout=True)


def _exec_notebook(notebook_filename):
    full_notebook_filename = os.path.join(_nbpath(), notebook_filename)
    with open(full_notebook_filename) as notebook_file:
        nb = nbformat.read(notebook_file, as_version=nbformat.current_nbformat)
        try:
            _preproc().preprocess(nb, {'metadata': {'path': _nbpath()}})
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
