from __future__ import print_function
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import glob
import traceback
import unittest
import sys
import time


class TestNotebooks(unittest.TestCase):
    def _execNotebook(self, ep, notebook_filename, nbpath):
        with open(notebook_filename) as f:
            nb = nbformat.read(f, as_version=nbformat.current_nbformat)
            try:
                out = ep.preprocess(nb, {'metadata': {'path': nbpath}})
            except CellExecutionError:
                print('-' * 60)
                traceback.print_exc(file=sys.stdout)
                print('-' * 60)
                self.assertFalse('Error executing the notebook %s. See above for error.' % notebook_filename)

    def test_all_notebooks(self):
        ''' Test all notebooks except blacklist. Blacklisted notebooks take too long.'''
        blacklist = ['svi_test.ipynb']
        pythonkernel = 'python'+str(sys.version_info[0])
        nbpath = '../doc/source/notebooks/'
        # see http://nbconvert.readthedocs.io/en/stable/execute_api.html
        ep = ExecutePreprocessor(timeout=120, kernel_name=pythonkernel, interrupt_on_timeout=True)
        lfiles = glob.glob(nbpath+"*.ipynb")
        for notebook_filename in lfiles:
            if(notebook_filename not in blacklist):
                t = time.time()
                self._execNotebook(ep, notebook_filename, nbpath)
                print(notebook_filename, 'took %g seconds.' % (time.time()-t))
if __name__ == '__main__':
    unittest.main()