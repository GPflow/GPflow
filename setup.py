#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import sys
from setuptools import setup, Extension
import numpy as np


#Mac OS X Clang doesn't support OpenMP at the current time.
#This detects if we are building on a Mac
def ismac():
    return sys.platform[:6] == 'darwin'

if ismac():
    compile_flags = [ '-O3', ]
    link_args = []
else:
    compile_flags = [ '-fopenmp', '-O3', ]
    link_args = ['-lgomp']

setup(name = 'GPflow',
      version = "0.0.1",
      author = "James Hensman, Alex Matthews",
      author_email = "james.hensman@gmail.com",
      description = ("Gaussian process methods in tensorflow"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels tensorflow",
      url = "none yet",
      ext_modules = [],
      packages = ["GPflow"],
      package_dir={'GPflow': 'GPflow'},
      py_modules = ['GPflow.__init__'],
      test_suite = 'testing',
      install_requires=['numpy>=1.9', 'scipy>=0.16'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
