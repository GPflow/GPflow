from __future__ import print_function
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import glob
import time
import sys

pythonkernel = 'python'+str(sys.version_info[0])
nbpath = '../doc/source/notebooks/'
# see http://nbconvert.readthedocs.io/en/stable/execute_api.html
ep = ExecutePreprocessor(timeout=5, kernel_name=pythonkernel, interrupt_on_timeout=True)
for notebook_filename in glob.glob(nbpath+"*.ipynb"):
    print(notebook_filename)
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        try:
            t = time.time()
            out = ep.preprocess(nb, {'metadata': {'path': nbpath}})
        except CellExecutionError:
            print('Error executing the notebook "%s".\n\n' % notebook_filename)
            raise
        print('Processing %s took %g seconds.' % (notebook_filename, time.time()-t))
