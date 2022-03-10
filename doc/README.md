# GPflow documentation

## Read documentation online

The documentation is stored in a special branch
[`gp-pages`](https://github.com/GPflow/GPflow/tree/gh-pages) and served by
[GitHub Pages](https://pages.github.com/).

We serve a version of documentation for the most recent `develop` branch and for all releases since
`2.4.0`. You can find them online here:

* Redirect to most recent release: https://gpflow.github.io/GPflow/
* `develop` https://gpflow.github.io/GPflow/develop

Normally our CircleCI build is responsible for building our documentation whenever there is a merge
to `develop` or `master`. See the configuration
[here](https://github.com/GPflow/GPflow/blob/develop/.circleci/config.yml).


## Compile documentation locally

To compile the GPflow documentation locally:

1. Change to the GPflow source directory.

2. Install dev dependencies
   ```bash
   make dev-install
   ```

   If pandoc does not install via pip, or step 4 does not work, go to pandoc.org/installing.html (the PyPI package depends on the external system-wide installation of pandoc executables)

3. Generate auto-generated files
   ```bash
   doc_build_dir="/tmp/gpflow_docs"
   python doc/build_docs.py develop ${doc_build_dir}
   ```

6. Check documentation locally by opening (in a browser) `${doc_build_dir}/develop/index.html`.


## Run notebooks locally

The notebooks underneath `source/notebooks` rely on [jupytext](https://github.com/mwouts/jupytext).
Make sure to [install the `jupytext` package](https://github.com/mwouts/jupytext#install) before
calling `jupyter notebook <notebook_file.pct.py>`
(which will automatically create the paired .ipynb file).
