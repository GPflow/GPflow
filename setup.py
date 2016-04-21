#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re


# load version form _version.py
VERSIONFILE = "GPflow/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='GPflow',
      version=verstr,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description=("Gaussian process methods in tensorflow"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels tensorflow",
      url = "http://github.com/gpflow/gpflow",
      ext_modules = [],
      packages = ["GPflow"],
      package_dir={'GPflow': 'GPflow'},
      py_modules = ['GPflow.__init__'],
      test_suite = 'testing',
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'tensorflow>=0.7.1'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
