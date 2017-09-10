#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

import re
import os
import sys
from pkg_resources import parse_version

# load version form _version.py
VERSIONFILE = "gpflow/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Dependencies of GPflow
dependencies = ['numpy>=1.9', 'scipy>=0.16']
min_tf_version = '1.0.0'

# Only detect TF if not installed or outdated. If not, do not do not list as 
# requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip). 
try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf
    if parse_version(tf.__version__) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning) as e:
    # Add TensorFlow to dependencies to trigger installation/update
    dependencies.append('tensorflow>={0}'.format(min_tf_version))

setup(name='gpflow',
      version=verstr,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description=("Gaussian process methods in tensorflow"),
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/gpflow/gpflow",
      package_data={'gpflow': ['gpflow/gpflowrc']},
      include_package_data=True,
      ext_modules=[],
      packages=["gpflow"],
      package_dir={'gpflow': 'gpflow'},
      py_modules=['gpflow.__init__'],
      test_suite='testing',
      install_requires=dependencies,
      extras_require={'tensorflow with gpu': ['tensorflow-gpu>=1.0.0'],
                      'Export parameters as pandas dataframes': ['pandas>=0.18.1']},
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'])
