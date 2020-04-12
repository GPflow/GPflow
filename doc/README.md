# GPflow documentation

## Read documentation online

[![Documentation Status (master)](https://readthedocs.org/projects/gpflow/badge/?version=master)](http://gpflow.readthedocs.io/en/master/?badge=master)
[![Documentation Status (develop)](https://readthedocs.org/projects/gpflow/badge/?version=develop)](http://gpflow.readthedocs.io/en/develop/?badge=develop)

We use readthedocs to build the online documentation, and have separate versions for the `master` and `develop` branches:
https://gpflow.readthedocs.io/en/master/ (for the latest official release e.g. on PyPI) and
https://gpflow.readthedocs.io/en/develop/ (for the latest cutting-edge code available from the develop branch on github).


## Compile documentation locally

To compile the GPflow documentation locally:

1. Change to this directory (e.g. `cd doc` if you are in the GPflow git repository's base directory)

2. Install doc dependencies
   ```
   pip install sphinx sphinx_rtd_theme numpydoc nbsphinx sphinx_autodoc_typehints ipython jupytext jupyter_client ipywidgets
   ```

3. Install pandoc
   ```
   pip install pandoc
   ```
   If pandoc does not install via pip, or step 5 does not work, go to pandoc.org/installing.html (the PyPI package depends on the external system-wide installation of pandoc executables)

4. Generate auto-generated files
   * Notebooks (.ipynb): run `make -C source/notebooks -j 4` (here with 4 parallel threads)
   * API documentation (.rst): run `python source/generate_module_rst.py`

5. Compile the documentation
   ```
   make html
   ```

6. Check documentation locally by opening (in a browser) build/html/index.html


## Setup for automatic documentation generation

Upon each merge to the `develop` branch, this repository's [CircleCI configuration](../.circleci/config.yml) runs the `trigger-docs-generation` step
which triggers a CircleCI build on the [GPflow/docs repository](https://github.com/GPflow/docs).
This clones the latest GPflow develop branch and compiles all notebooks from jupytext to .ipynb
(setting the `DOCS` environment variable so that notebook optimisations are run to convergence)
and runs the `generate_module_rst.py` script as above to generate the .rst files for API documentation.
(This script is run on CircleCI, instead of ReadTheDocs, as it requires gpflow and hence `tensorflow` and `tensorflow_probability` to be installed, but TensorFlow is too large to be installed inside the ReadTheDocs docker images.)
ReadTheDocs then pulls in these auto-generated files as well as all other files within this doc/ directory to actually build the documentation using Sphinx.
