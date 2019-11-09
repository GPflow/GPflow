#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

import sys
from pathlib import Path

from pkg_resources import parse_version
from setuptools import find_packages, setup

is_py37 = sys.version_info.major == 3 and sys.version_info.minor == 7

# Dependencies of GPflow
requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'multipledispatch>=0.4.9',
    'tensorflow-probability>=0.8',
    'tabulate',
] + ([] if is_py37 else ["dataclasses"])

min_tf_version = '2.0.0'
tf_cpu = 'tensorflow'
tf_gpu = 'tensorflow-gpu'

# Only detect TF if not installed or outdated. If not, do not do not list as
# requirement to avoid installing over e.g. tensorflow-gpu
# To avoid this, rely on importing rather than the package name (like pip).

try:
    # If tf not installed, import raises ImportError
    import tensorflow as tf
    if parse_version(tf.__version__) < parse_version(min_tf_version):
        # TF pre-installed, but below the minimum required version
        raise DeprecationWarning("TensorFlow version below minimum requirement")
except (ImportError, DeprecationWarning):
    # Add TensorFlow to dependencies to trigger installation/update
    requirements.append(tf_cpu)

with open(str(Path(".", "VERSION").absolute())) as version_file:
    version = version_file.read().strip()

packages = find_packages('.', exclude=["tests"])

setup(name='gpflow',
      version=version,
      author="James Hensman, Alex Matthews",
      author_email="james.hensman@gmail.com",
      description="Gaussian process methods in TensorFlow",
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/GPflow/GPflow",
      packages=packages,
      include_package_data=True,
      install_requires=requirements,
      extras_require={'Tensorflow with GPU': [tf_gpu]},
      python_requires=">=3.6",
      classifiers=[
          'License :: OSI Approved :: Apache Software License',
          'Natural Language :: English',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Artificial Intelligence'
      ])
