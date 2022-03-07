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
        "tensorflow>=2.4.0",
        "tensorflow-probability>=0.12.0",
        # NOTE: once we require tensorflow-probability>=0.12, we can remove our custom deepcopy handling
        "setuptools>=41.0.0",  # to satisfy dependency constraints
    ]

else:
    requirements = []

requirements.extend(
    [
        "numpy",
        "scipy",
        "multipledispatch>=0.6",
        "tabulate",
        "typing_extensions",
        "packaging",
        "deprecated",
        "lark>=1.1.0",
    ]
)


def read_file(filename: str) -> str:
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
    package_data={"": ["*.lark"]},
    include_package_data=True,
    install_requires=requirements,
    extras_require={"ImageToTensorBoard": ["matplotlib"]},
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Environment :: GPU :: NVIDIA CUDA",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Typing :: Typed",
    ],
)
