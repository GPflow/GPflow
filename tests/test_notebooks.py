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

import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError

from gpflow.test_util import GPflowTestCase


class TestNotebooks(GPflowTestCase):
    """
    Run notebook tests.

    Blacklist:
    - svi.ipynb
    - GPLVM.ipynb
    - regression.ipynb
    """
    nbpath = None
    preproc = None

    @classmethod
    def setUpClass(cls):
        pythonkernel = 'python' + str(sys.version_info[0])
        this_dir = os.path.dirname(__file__)
        cls.nbpath = os.path.join(this_dir, '../doc/source/notebooks/')
        cls.preproc = ExecutePreprocessor(
            timeout=120, kernel_name=pythonkernel, interrupt_on_timeout=True)

    def _exec_notebook(self, notebook_filename):
        full_notebook_filename = os.path.join(self.nbpath, notebook_filename)
        with open(full_notebook_filename) as notebook_file:
            nb = nbformat.read(notebook_file, as_version=nbformat.current_nbformat)
            try:
                self.preproc.preprocess(nb, {'metadata': {'path': self.nbpath}})
            except CellExecutionError as cell_error:
                print('-' * 60)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60)
                msg = 'Error executing the notebook {0}. See above for error.\nCell error: {1}'
                self.fail(msg.format(notebook_filename, str(cell_error)))

    def _exec_notebook_ts(self, notebook_filename):
        with self.test_context():
            ts = time.time()
            self._exec_notebook(notebook_filename)
            print(notebook_filename, 'took {0} seconds.'.format(time.time() - ts))

    # def test_classification(self):
    #     self._exec_notebook_ts("classification.ipynb")

    # def test_coreg_demo(self):
    #     self._exec_notebook_ts("coreg_demo.ipynb")

    # def test_kernels(self):
    #     self._exec_notebook_ts("kernels.ipynb")

    # def test_mcmc(self):
    #     self._exec_notebook_ts("mcmc.ipynb")

    # def test_ordinal(self):
    #     self._exec_notebook_ts("ordinal.ipynb")

    # def test_sanity_check(self):
    #     self._exec_notebook_ts("Sanity_check.ipynb")

    # def test_settings(self):
    #     self._exec_notebook_ts("settings.ipynb")

    # def test_SGPR_notes(self):
    #     self._exec_notebook_ts("SGPR_notes.ipynb")

    # def test_vgp_notes(self):
    #     self._exec_notebook_ts("vgp_notes.ipynb")

    # TODO(@awav): CHECK IT
    # def FITCvsVFE(self):
    #    self._exec_notebook_ts("FITCvsVFE.ipynb")

    # TODO(@awav): CHECK IT
    # def models(self):
    #     self._exec_notebook_ts("models.ipynb")

    # TODO(@awav): CHECK IT
    # def multiclass(self):
    #     self._exec_notebook_ts("multiclass.ipynb")
