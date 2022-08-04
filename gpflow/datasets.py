""" Modified from 'hughsalimbeni/bayesian_benchmarks.git'

Load and pre-process a range of UCI datasets. If dataset already exists in
DATASET_DIR it will skip downloading it.

Usage:

data = get_regression_data('boston')

X_train, Y_train, X_test, Y_test = data.X_train, data.Y_train, data.X_test, data.Y_test
N, D = data.N, data.D
"""

import logging
import os
from urllib.request import urlopen

import numpy as np
import pandas

logging.getLogger().setLevel(logging.INFO)
import zipfile

DATASET_DIR = "/home/fergus/datasets"
BASE_SEED = 42
uci_base_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/"

_ALL_REGRESSION_DATATSETS = {}


def normalize(X):
    X_mean = np.average(X, 0)[None, :]
    X_std = 1e-6 + np.std(X, 0)[None, :]
    return (X - X_mean) / X_std, X_mean, X_std


class Dataset(object):
    def __init__(self, split=0, prop=0.9):
        if self.needs_download:
            self.download()

        X_raw, Y_raw = self.read_data()
        self.X, self.Y = self.preprocess_data(X_raw, Y_raw)

        ind = np.arange(self.N)

        self.random = np.random.RandomState(BASE_SEED + split)
        self.random.shuffle(ind)

        self.n_train = int(self.N * prop)

        self.X_train = self.X[ind[: self.n_train]]
        self.Y_train = self.Y[ind[: self.n_train]]

        self.X_test = self.X[ind[self.n_train :]]
        self.Y_test = self.Y[ind[self.n_train :]]

    def apply_split(self, split: int):
        ind = np.arange(self.N)

        self.random = np.random.RandomState(BASE_SEED + split)
        self.random.shuffle(ind)

        self.X_train = self.X[ind[: self.n_train]]
        self.Y_train = self.Y[ind[: self.n_train]]

        self.X_test = self.X[ind[self.n_train :]]
        self.Y_test = self.Y[ind[self.n_train :]]

    @property
    def datadir(self):
        dir = os.path.join(DATASET_DIR, self.name)
        if not os.path.isdir(dir):
            os.makedirs(dir)
        return dir

    @property
    def datapath(self):
        # print(self.url)
        filename = self.url.split("/")[-1]  # this is for the simple case with no zipped files
        return os.path.join(self.datadir, filename)

    @property
    def needs_download(self):
        return not os.path.isfile(self.datapath)

    def download(self):
        logging.info("downloading {} data".format(self.name))

        is_zipped = np.any([z in self.url for z in [".gz", ".zip", ".tar"]])

        if is_zipped:
            filename = os.path.join(self.datadir, self.url.split("/")[-1])
        else:
            filename = self.datapath

        with urlopen(self.url) as response, open(filename, "wb") as out_file:
            data = response.read()
            out_file.write(data)

        if is_zipped:
            zip_ref = zipfile.ZipFile(filename, "r")
            zip_ref.extractall(self.datadir)
            zip_ref.close()

            # os.remove(filename)

        logging.info("finished donwloading {} data".format(self.name))

    def read_data(self):
        raise NotImplementedError

    def preprocess_data(self, X, Y):
        X, self.X_mean, self.X_std = normalize(X)
        Y, self.Y_mean, self.Y_std = normalize(Y)
        return X, Y

    @property
    def train(self):
        return (self.X_train, self.Y_train)

    @property
    def test(self):
        return (self.X_test, self.Y_test)

    @property
    def sparse_only(self):
        return self.N > 5_000


def add_regression(C):
    _ALL_REGRESSION_DATATSETS.update({C.name: C})
    return C


@add_regression
class Boston(Dataset):
    N, D, name = 506, 13, "boston"
    url = uci_base_url + "housing/housing.data"

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Concrete(Dataset):
    N, D, name = 1030, 8, "concrete"
    url = uci_base_url + "concrete/compressive/Concrete_Data.xls"

    def read_data(self):
        data = pandas.read_excel(self.datapath).values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Energy(Dataset):
    N, D, name = 768, 8, "energy"
    url = uci_base_url + "00242/ENB2012_data.xlsx"

    def read_data(self):
        # NB this is the first output (aka Energy1, as opposed to Energy2)
        data = (
            pandas.read_excel(self.datapath, engine="openpyxl", usecols=np.arange(9))
            .dropna()
            .values
        )
        assert len(data) == self.N
        return data[:, :-1], data[:, -1].reshape(-1, 1)


# @add_regression
# class Kin8mn(Dataset):
#     N, D, name = 8192, 8, 'kin8mn'
#     url = 'http://mldata.org/repository/data/download/csv/uci-20070111-kin8nm'
#     def read_data(self):
#         data = pandas.read_csv(self.datapath, header=None).values
#         return data[:, :-1], data[:, -1].reshape(-1, 1)


@add_regression
class Naval(Dataset):
    N, D, name = 11934, 14, "naval"
    url = uci_base_url + "00316/UCI%20CBM%20Dataset.zip"

    @property
    def datapath(self):
        return os.path.join(self.datadir, "UCI CBM Dataset/data.txt")

    def read_data(self):
        data = pandas.read_fwf(self.datapath, header=None).values
        # NB this is the first output
        X = data[:, :-2]
        Y = data[:, -2].reshape(-1, 1)

        # dims 8 and 11 have std=0:
        X = np.delete(X, [8, 11], axis=1)
        return X, Y


@add_regression
class Power(Dataset):
    N, D, name = 9568, 4, "power"
    url = uci_base_url + "00294/CCPP.zip"

    @property
    def datapath(self):
        return os.path.join(self.datadir, "CCPP/Folds5x2_pp.xlsx")

    def read_data(self):
        data = pandas.read_excel(self.datapath, engine="openpyxl").values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


# @add_regression
# class Protein(Dataset):
#     N, D, name = 45730, 9, "protein"
#     url = uci_base_url + "00265/CASP.csv"
#
#     def read_data(self):
#         data = pandas.read_csv(self.datapath).values
#         return data[:, 1:], data[:, 0].reshape(-1, 1)


@add_regression
class WineRed(Dataset):
    N, D, name = 1599, 11, "winered"
    url = uci_base_url + "wine-quality/winequality-red.csv"

    def read_data(self):
        data = pandas.read_csv(self.datapath, delimiter=";").values
        return data[:, :-1], data[:, -1].reshape(-1, 1)


# @add_regression
# class WineWhite(WineRed):
#     N, D, name = 4898, 11, "winewhite"
#     url = uci_base_url + "wine-quality/winequality-white.csv"


@add_regression
class Yacht(Dataset):
    N, D, name = 308, 6, "yacht"
    url = uci_base_url + "/00243/yacht_hydrodynamics.data"

    def read_data(self):
        data = np.loadtxt(self.datapath)
        return data[:, :-1], data[:, -1].reshape(-1, 1)


regression_datasets = list(_ALL_REGRESSION_DATATSETS.keys())
regression_datasets.sort()

#todo remove this hack for selecting single dataset only
regression_datasets = [regression_datasets[-2]]


def get_regression_data(name, *args, **kwargs):
    return _ALL_REGRESSION_DATATSETS[name](*args, **kwargs)
