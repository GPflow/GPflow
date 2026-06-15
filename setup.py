#!/usr/bin/env python
# -*- coding: utf-8 -*-

# pylint: skip-file

from typing import Any

from setuptools import find_packages, setup

##### Dependencies of GPflow

requirements = [
    "importlib_metadata; python_version<'3.8'",
    "check_shapes>=1.0.0",
    "deprecated",
    "multipledispatch>=0.6",
    "numpy<2",
    "packaging",
    "scipy",
    "setuptools>=41.0.0",  # to satisfy dependency constraints
    "tabulate",
    "tensorflow-probability[tf]>=0.12.0",
    "tensorflow>=2.4.0; platform_system!='Darwin' or platform_machine!='arm64'",
    # NOTE: Support of Apple Silicon MacOS platforms is in an experimental mode
    "tensorflow-macos>=2.4.0; platform_system=='Darwin' and platform_machine=='arm64'",
    # NOTE: once we require tensorflow-probability>=0.12, we can remove our custom deepcopy handling
    "typing_extensions",
]


def read_file(filename: str) -> str:
    with open(filename, encoding="utf-8") as f:
        return f.read().strip()


version = read_file("VERSION")
readme_text = read_file("README.md")

packages = find_packages(".", exclude=["tests"])

# Mypy + types-setuptools on Python 3.10+ reject plain dicts for these kwargs.
_project_urls: Any = {
    "Source on GitHub": "https://github.com/GPflow/GPflow",
    "Documentation": "https://gpflow.github.io/GPflow/",
}
_package_data: Any = {"": ["*.lark"]}
_extras_require: Any = {"ImageToTensorBoard": ["matplotlib"]}

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
    project_urls=_project_urls,
    packages=packages,
    package_data=_package_data,
    include_package_data=True,
    install_requires=requirements,
    extras_require=_extras_require,
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
