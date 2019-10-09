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
import traceback

import nbformat
import pytest
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

# To blacklist a notebook, add its full base name (including .ipynb extension,
# but without any directory component). If there are several notebooks in
# different directories with the same base name, they will all get blacklisted
# (change the blacklisting check to something else in that case, if need be!)
BLACKLISTED_NOTEBOOKS = [
    "kernel_design.ipynb",  # PR #1087
    "regression.ipynb",  # PR #1076
    "varying_noise.ipynb",  # PR #1050
    "Sanity_check.ipynb",  # PR #1078

    "external-mean-function.ipynb",  # TODO: @vdutor
    "natural_gradients.ipynb",  # TODO: @awav
    "multiclass_classification.ipynb",  # TODO @jordigraumo
    "mcmc.ipynb",  # TODO: @condnsdmatters

    "models.ipynb",
    "gp_nn.ipynb",
    "mixture_density_network.ipynb",

    "upper_bound.ipynb",
    "FITCvsVFE.ipynb",

    "monitoring.ipynb",  # requires re-write for new way of monitoring
    "settings.ipynb",  # requires re-write for new config

    "tips_and_tricks.ipynb",  # requires a big re-write but contains some useful
    # sections such as saving&loading...

    "tf_graphs_and_sessions.ipynb",  # to be deleted - only tf1 graph&sessions
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
    return ExecutePreprocessor(timeout=300, kernel_name=pythonkernel, interrupt_on_timeout=True)


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


@pytest.mark.notebooks
@pytest.mark.parametrize('notebook_file', get_notebooks())
def test_notebook(notebook_file):
    _exec_notebook(notebook_file)
