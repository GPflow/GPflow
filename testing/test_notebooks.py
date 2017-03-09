import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from nbconvert.preprocessors.execute import CellExecutionError
import glob
import time
import sys

#TODO: how to do black list?
pythonkernel = 'python'+str(sys.version_info[0])
nbpath = '../doc/source/notebooks/'
# see http://nbconvert.readthedocs.io/en/stable/execute_api.html
ep = ExecutePreprocessor(timeout=5, kernel_name=pythonkernel, interrupt_on_timeout=True)
for file in glob.glob(nbpath+"*.ipynb"):
    notebook_filename = file
    print(file)
    with open(notebook_filename) as f:
        nb = nbformat.read(f, as_version=4)
        try:
            t = time.time()
            out = ep.preprocess(nb, {'metadata': {'path': nbpath}})
        except CellExecutionError:
            msg = 'Error executing the notebook "%s".\n\n' % notebook_filename
            print(msg)
            raise
        print('Processing %s took %g seconds.' % (notebook_filename, time.time()-t))
