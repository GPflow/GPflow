# Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Definitions of our datasets.

Code adapted from:
https://github.com/hughsalimbeni/bayesian_benchmarks/blob/master/bayesian_benchmarks/data.py
"""
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from benchmark.dataset_api import (
    LARGE,
    MEDIUM,
    REAL_DATA,
    REGRESSION,
    SYNTHETIC,
    TINY,
    make_unsplit_array_dataset_factory,
    make_url_dataset_factory,
)
from gpflow.base import AnyNDArray


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, TINY, SYNTHETIC},
    test_fraction=0.5,
    normalise=False,
)
def tiny_linear(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    X = rng.random((20, 1))
    F = 0.3 * X - 0.1
    noise = rng.normal(0.0, 0.1, F.shape)
    Y = F + noise
    return X, Y


@make_unsplit_array_dataset_factory(
    tags={REGRESSION, TINY, SYNTHETIC},
    test_fraction=0.5,
    normalise=False,
)
def tiny_sine(rng: np.random.Generator) -> Tuple[AnyNDArray, AnyNDArray]:
    X = rng.random((20, 1))
    F = np.sin(10 * X)
    noise = rng.normal(0.0, 0.1, F.shape)
    Y = F + noise
    return X, Y


UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/"


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "housing/housing.data",
)
def boston(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "concrete/compressive/Concrete_Data.xls",
)
def concrete(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_excel(path).values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "00242/ENB2012_data.xlsx",
)
def energy(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_excel(path, engine="openpyxl", usecols=np.arange(9)).dropna().values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, LARGE, REAL_DATA},
    url=UCI_BASE_URL + "00316/UCI%20CBM%20Dataset.zip",
    archive_member=Path("UCI CBM Dataset/data.txt"),
)
def naval(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_fwf(path, header=None).values
    # NB this is the first output
    X = data[:, :-2]
    Y = data[:, -2].reshape(-1, 1)

    # dims 8 and 11 have std=0:
    X = np.delete(X, [8, 11], axis=1)
    return X, Y


@make_url_dataset_factory(
    tags={REGRESSION, LARGE, REAL_DATA},
    url=UCI_BASE_URL + "00294/CCPP.zip",
    archive_member=Path("CCPP/Folds5x2_pp.xlsx"),
)
def power(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_excel(path, engine="openpyxl").values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, LARGE, REAL_DATA},
    url=UCI_BASE_URL + "00265/CASP.csv",
)
def protein(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_csv(path).values
    return data[:, 1:], data[:, 0].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "wine-quality/winequality-red.csv",
)
def red_wine(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_csv(path, delimiter=";").values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, LARGE, REAL_DATA},
    url=UCI_BASE_URL + "wine-quality/winequality-white.csv",
)
def white_wine(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = pd.read_csv(path, delimiter=";").values
    return data[:, :-1], data[:, -1].reshape(-1, 1)


@make_url_dataset_factory(
    tags={REGRESSION, MEDIUM, REAL_DATA},
    url=UCI_BASE_URL + "/00243/yacht_hydrodynamics.data",
)
def yacht(path: Path) -> Tuple[AnyNDArray, AnyNDArray]:
    data = np.loadtxt(path)
    return data[:, :-1], data[:, -1].reshape(-1, 1)
