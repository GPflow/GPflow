#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from setuptools import setup
from setuptools import find_packages

import re
import os
import sys
from pkg_resources import parse_version

# load version form _version.py
exec(open("gpflow/_version.py").read())

# Dependencies of GPflow
requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'pandas>=0.18.1',
    'multipledispatch>=0.4.9',
    'pytest>=3.5.0',
    'h5py>=2.7.0',
    'matplotlib>=2.2.2'
]

min_tf_version = '1.5.0'
tf_cpu = 'tensorflow>={}'.format(min_tf_version)
tf_gpu = 'tensorflow-gpu>={}'.format(min_tf_version)

# Only detect TF if not installed or outdated. If not, do not do not list as
# requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip).

try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf
    if parse_version(tf.VERSION) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning) as e:
    # Add TensorFlow to dependencies to trigger installation/update
    requirements.append(tf_cpu)

packages = find_packages('.')
package_data={'gpflow': ['gpflow/gpflowrc']}

setup(name='gpflow',
      version=__version__,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description=("Gaussian process methods in tensorflow"),
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/GPflow/GPflow",
      packages=packages,
      install_requires=requirements,
      package_data=package_data,
      include_package_data=True,
      test_suite='tests',
      extras_require={'Tensorflow with GPU': [tf_gpu]},
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
