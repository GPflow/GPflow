#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
from setuptools import setup
import re
import os
import sys

# load version form _version.py
VERSIONFILE = "GPflow/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

# Compile the bespoke TensorFlow ops in-place. Not sure how this would work if this script wasn't executed as `develop`.
compile_command = "g++ -std=c++11 -shared ./GPflow/tfops/vec_to_tri.cc " \
                  "GPflow/tfops/tri_to_vec.cc -o GPflow/tfops/matpackops.so " \
                  "-fPIC -I $(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')"
if sys.platform == "darwin":
    # Additional command for Macs, as instructed by the TensorFlow docs
    compile_command += " -undefined dynamic_lookup"
elif sys.platform.startswith("linux"):
    gcc_version = int(re.search('\d+.', os.popen("gcc --version").read()).group()[0])
    if gcc_version == 5:
        compile_command += " -D_GLIBCXX_USE_CXX11_ABI=0"
os.system(compile_command)

setup(name='GPflow',
      version=verstr,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description=("Gaussian process methods in tensorflow"),
      license="BSD 3-clause",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/gpflow/gpflow",
      package_data={'GPflow': ['GPflow/tfops/*.so', 'GPflow/gpflowrc']},
      include_package_data=True,
      ext_modules=[],
      packages=["GPflow"],
      package_dir={'GPflow': 'GPflow'},
      py_modules=['GPflow.__init__'],
      test_suite='testing',
      install_requires=['numpy>=1.9', 'scipy>=0.16', 'tensorflow>=0.11rc0'],
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence']
      )
