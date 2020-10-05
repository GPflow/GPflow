#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

import os
import sys

from setuptools import find_packages, setup

##### Dependencies of GPflow

# We do not want to install tensorflow in the readthedocs environment, where we
# use autodoc_mock_imports instead. Hence we use this flag to decide whether or
# not to append tensorflow and tensorflow_probability to the requirements:
if os.environ.get("READTHEDOCS") != "True":
    requirements = [
        "tensorflow>=2.1.0",
        "tensorflow-probability>0.10.0",  # tensorflow-probability==0.10.0 doesn't install correctly, https://github.com/tensorflow/probability/issues/991
        "setuptools>=41.0.0",  # to satisfy dependency constraints
    ]

else:
    requirements = []

requirements.extend(
    ["numpy", "scipy", "multipledispatch>=0.6", "tabulate", "typing_extensions", "packaging"]
)

if sys.version_info < (3, 7):
    requirements.append("dataclasses")  # became part of stdlib in python 3.7


def read_file(filename):
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


version = read_file("VERSION")
readme_text = read_file("README.md")

packages = find_packages(".", exclude=["tests"])

setup(
    name="gpflow",
    version=version,
    author="James Hensman, Alex Matthews",
    author_email="james.hensman@gmail.com",
    description="Gaussian process methods in TensorFlow",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    keywords="machine-learning gaussian-processes kernels tensorflow",
    url="https://www.gpflow.org",
    project_urls={
        "Source on GitHub": "https://github.com/GPflow/GPflow",
        "Documentation": "https://gpflow.readthedocs.io",
    },
    packages=packages,
    include_package_data=True,
    install_requires=requirements,
    extras_require={"ImageToTensorBoard": ["matplotlib"]},
    python_requires=">=3.6",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
