import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.features.mo_features as mf
import gpflow.kernels.mo_kernels as mk
from gpflow.covariances import mo_kufs, mo_kuus

rng = np.random.RandomState(9911)

# ------------------------------------------
# Helpers
# ------------------------------------------


def make_kernel():
    return gpflow.kernels.RBF()


def make_kernels(num):
    return [make_kernel() for _ in range(num)]


def make_ip():
    x = rng.permutation(Datum.X)
    return gpflow.features.InducingPoints(x[:Datum.M, ...])


def make_ips(num):
    return [make_ip() for _ in range(num)]


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Datum:
    D = 1
    L = 2
    P = 3
    M = 10
    N = 100
    W = rng.randn(P, L)
    X = rng.randn(N)[:, None]
    Xnew = rng.randn(N)[:, None]


multioutput_feature_list = [
    mf.SharedIndependentMof(make_ip()),
    mf.SeparateIndependentMof(make_ips(Datum.P))
]

multioutput_kernel_list = [
    mk.SharedIndependentMok(make_kernel(), Datum.P),
    mk.SeparateIndependentMok(make_kernels(Datum.L)),
    mk.SeparateMixedMok(make_kernels(Datum.L), Datum.W)
]


@pytest.mark.parametrize('feature', multioutput_feature_list)
@pytest.mark.parametrize('kernel', multioutput_kernel_list)
def test_kuu(feature, kernel):
    Kuu = mo_kuus.Kuu(feature, kernel, jitter=1e-9)
    tf.linalg.cholesky(Kuu)


@pytest.mark.parametrize('feature', multioutput_feature_list)
@pytest.mark.parametrize('kernel', multioutput_kernel_list)
def test_kuf(feature, kernel):
    Kuf = mo_kufs.Kuf(feature, kernel, Datum.Xnew)


@pytest.mark.parametrize('fun', [mo_kuus.Kuu, mo_kufs.Kuf])
def test_mixed_shared(fun):
    features = mf.MixedKernelSharedMof(make_ip())
    kernel = mk.SeparateMixedMok(make_kernels(Datum.L), Datum.W)
    if fun is mo_kuus.Kuu:
        t = tf.linalg.cholesky(fun(features, kernel, jitter=1e-9))
    else:
        t = fun(features, kernel, Datum.Xnew)
        print(t.shape)
