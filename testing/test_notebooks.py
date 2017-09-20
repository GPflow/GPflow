from __future__ import print_function

import traceback
import unittest
import sys
import time
import os

import nbformat

from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
from nose.plugins.attrib import attr
from testing.gpflow_testcase import GPflowTestCase

@attr(speed='slow')
class TestNotebooks(GPflowTestCase):
    """
    Run notebook tests.

    Blacklist:
    - svi_test.ipynb
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
        with self.test_session():
            ts = time.time()
            self._exec_notebook(notebook_filename)
            print(notebook_filename, 'took {0} seconds.'.format(time.time() - ts))

    def classification_test(self):
        self._exec_notebook_ts("classification.ipynb")

    def coreg_demo_test(self):
        self._exec_notebook_ts("coreg_demo.ipynb")

    def FITCvsVFE_test(self):
        self._exec_notebook_ts("FITCvsVFE.ipynb")

    def kernels_test(self):
        self._exec_notebook_ts("kernels.ipynb")

    def mcmc_test(self):
        self._exec_notebook_ts("mcmc.ipynb")

    def models_test(self):
        self._exec_notebook_ts("models.ipynb")

    def multiclass_test(self):
        self._exec_notebook_ts("multiclass.ipynb")

    def ordinal_test(self):
        self._exec_notebook_ts("ordinal.ipynb")

    def regression_with_updated_data_test(self):
        self._exec_notebook_ts("regression_with_updated_data.ipynb")

    def sanity_check_test(self):
        self._exec_notebook_ts("Sanity_check.ipynb")

    def settings_test(self):
        self._exec_notebook_ts("settings.ipynb")

    def SGPR_notes_test(self):
        self._exec_notebook_ts("SGPR_notes.ipynb")

    def structure_test(self):
        self._exec_notebook_ts("structure.ipynb")

    def vgp_notes_test(self):
        self._exec_notebook_ts("vgp_notes.ipynb")

if __name__ == '__main__':
    unittest.main()
