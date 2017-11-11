#!/usr/bin/env python
# -*- coding: utf-8 -*-

#  pylint: skip-file

from setuptools import setup
from setuptools import find_packages

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
dependencies = ['numpy>=1.10.0', 'scipy>=0.18.0', 'pandas>=0.18.1']
min_tf_version = '1.3.0'

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

packages = find_packages('.')
package_data={'gpflow': ['gpflow/gpflowrc']}

setup(name='gpflow',
      version=verstr,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description=("Gaussian process methods in tensorflow"),
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/GPflow/GPflow",
      packages=packages,
      install_requires=dependencies,
      tests_require=['pytest'],
      package_data=package_data,
      include_package_data=True,
      test_suite='tests',
      extras_require={'Tensorflow with GPU': ['tensorflow-gpu>=1.3.0'],
                      'Export parameters as pandas dataframes': ['pandas>=0.18.1']},
      classifiers=['License :: OSI Approved :: Apache Software License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Programming Language :: Python :: 3.5',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'])
