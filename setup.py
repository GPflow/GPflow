#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from setuptools import setup

import re
import os
import sys
import tensorflow as tf

# load version form _version.py
VERSIONFILE = "gpflow/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

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
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'pandas>=0.18.1'],
      tests_require=['matplotlib'],
      extras_require={'tensorflow': ['tensorflow>=1.0.0'],
                      'tensorflow with gpu': ['tensorflow-gpu>=1.0.0']},
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
