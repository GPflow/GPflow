#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

import os
from pathlib import Path

from pkg_resources import parse_version
from setuptools import find_packages, setup

# Dependencies of GPflow
requirements = [
    'numpy>=1.10.0',
    'scipy>=0.18.0',
    'pandas>=0.18.1',
    'multipledispatch>=0.4.9',
    'pytest>=3.5.0',
    'h5py>=2.7.0',
    'matplotlib>=2.2.2',
    'tfp-nightly'
]

min_tf_version = '2.0.0'
tf_cpu = 'tf-nightly-2.0-preview'
tf_gpu = 'tf-nightly-gpu-2.0-preview'

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
      description=("Gaussian process methods in TensorFlow"),
      license="Apache License 2.0",
      keywords="machine-learning gaussian-processes kernels tensorflow",
      url="http://github.com/GPflow/GPflow",
      packages=packages,
      install_requires=requirements,
      include_package_data=True,
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
