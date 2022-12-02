# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from pathlib import Path
from typing import Sequence

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = "GPflow"
copyright = "2022, The GPflow Contributors"
author = "The GPflow Contributors"

# The full version, including alpha/beta/rc tags
release = "2.6.4"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.bibtex",
    "nbsphinx",
    "sphinx_autodoc_typehints",
    "IPython.sphinxext.ipython_console_highlighting",
]


# Configuration for highlighting in code examples:
highlight_options = {"stripall": True}

set_type_checking_flag = True
typehints_fully_qualified = False
always_document_param_types = True
# autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns: Sequence[str] = []


bibtex_bibfiles = ["refs.bib"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "pydata_sphinx_theme"
html_logo = "_static/gpflow_logo.svg"
html_css_files = ["pydata-custom.css"]

# theme-specific options. see theme docs for more info
html_theme_options = {
    "show_prev_next": False,
    "github_url": "https://github.com/GPflow/GPflow",
    "switcher": {
        "json_url": "https://gpflow.github.io/GPflow/versions.json",
        "version_match": Path("build_version.txt").read_text().strip(),
    },
    "navbar_end": ["version-switcher"],
}

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False  # Remove 'view source code' from top of page (for html, not python)

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
