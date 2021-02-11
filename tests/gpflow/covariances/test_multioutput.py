import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow.covariances.multioutput import kufs as mo_kufs
from gpflow.covariances.multioutput import kuus as mo_kuus

rng = np.random.RandomState(9911)

# ------------------------------------------
# Helpers
# ------------------------------------------


def make_kernel():
    return gpflow.kernels.SquaredExponential()


def make_kernels(num):
    return [make_kernel() for _ in range(num)]


def make_ip():
    X = rng.permutation(Datum.X)
    return gpflow.inducing_variables.InducingPoints(X[: Datum.M, ...])


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


multioutput_inducing_variable_list = [
    mf.SharedIndependentInducingVariables(make_ip()),
    mf.SeparateIndependentInducingVariables(make_ips(Datum.P)),
]

multioutput_kernel_list = [
    mk.SharedIndependent(make_kernel(), Datum.P),
    mk.SeparateIndependent(make_kernels(Datum.L)),
    mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W),
]


@pytest.mark.parametrize("inducing_variable", multioutput_inducing_variable_list)
@pytest.mark.parametrize("kernel", multioutput_kernel_list)
def test_kuu(inducing_variable, kernel):
    Kuu = mo_kuus.Kuu(inducing_variable, kernel, jitter=1e-9)
    tf.linalg.cholesky(Kuu)


@pytest.mark.parametrize("inducing_variable", multioutput_inducing_variable_list)
@pytest.mark.parametrize("kernel", multioutput_kernel_list)
def test_kuf(inducing_variable, kernel):
    Kuf = mo_kufs.Kuf(inducing_variable, kernel, Datum.Xnew)


@pytest.mark.parametrize("fun", [mo_kuus.Kuu, mo_kufs.Kuf])
def test_mixed_shared(fun):
    inducing_variable = mf.SharedIndependentInducingVariables(make_ip())
    kernel = mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W)
    if fun is mo_kuus.Kuu:
        t = tf.linalg.cholesky(fun(inducing_variable, kernel, jitter=1e-9))
    else:
        t = fun(inducing_variable, kernel, Datum.Xnew)
        print(t.shape)
