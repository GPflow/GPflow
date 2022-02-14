from typing import Callable, Sequence

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow.covariances.multioutput import kufs as mo_kufs
from gpflow.covariances.multioutput import kuus as mo_kuus
from gpflow.inducing_variables import InducingVariables
from gpflow.kernels import Kernel

rng = np.random.RandomState(9911)

# ------------------------------------------
# Helpers
# ------------------------------------------


def make_kernel() -> Kernel:
    return gpflow.kernels.SquaredExponential()


def make_kernels(num: int) -> Sequence[Kernel]:
    return [make_kernel() for _ in range(num)]


def make_ip() -> InducingVariables:
    X = rng.permutation(Datum.X)
    return gpflow.inducing_variables.InducingPoints(X[: Datum.M, ...])


def make_ips(num: int) -> Sequence[InducingVariables]:
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

multioutput_fallback_inducing_variable_list = [
    mf.FallbackSharedIndependentInducingVariables(make_ip()),
    mf.FallbackSeparateIndependentInducingVariables(make_ips(Datum.P)),
]

multioutput_kernel_list = [
    mk.SharedIndependent(make_kernel(), Datum.P),
    mk.SeparateIndependent(make_kernels(Datum.L)),
    mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W),
]


@pytest.mark.parametrize("inducing_variable", multioutput_inducing_variable_list)
@pytest.mark.parametrize("kernel", multioutput_kernel_list)
def test_kuu_shape(inducing_variable: InducingVariables, kernel: Kernel) -> None:
    Kuu = mo_kuus.Kuu(inducing_variable, kernel, jitter=1e-9)
    t = tf.linalg.cholesky(Kuu)

    if isinstance(kernel, mk.SharedIndependent):
        if isinstance(inducing_variable, mf.SeparateIndependentInducingVariables):
            assert t.shape == (3, 10, 10)
        else:
            assert t.shape == (10, 10)
    else:
        assert t.shape == (2, 10, 10)


@pytest.mark.parametrize("inducing_variable", multioutput_inducing_variable_list)
@pytest.mark.parametrize("kernel", multioutput_kernel_list)
def test_kuf_shape(inducing_variable: InducingVariables, kernel: Kernel) -> None:
    Kuf = mo_kufs.Kuf(inducing_variable, kernel, Datum.Xnew)

    if isinstance(kernel, mk.SharedIndependent):
        if isinstance(inducing_variable, mf.SeparateIndependentInducingVariables):
            assert Kuf.shape == (3, 10, 100)
        else:
            assert Kuf.shape == (10, 100)
    else:
        assert Kuf.shape == (2, 10, 100)


@pytest.mark.parametrize("inducing_variable", multioutput_fallback_inducing_variable_list)
def test_kuf_fallback_shared_inducing_variables_shape(inducing_variable: InducingVariables) -> None:
    kernel = mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W)
    Kuf = mo_kufs.Kuf(inducing_variable, kernel, Datum.Xnew)

    assert Kuf.shape == (10, 2, 100, 3)


@pytest.mark.parametrize("fun", [mo_kuus.Kuu, mo_kufs.Kuf])
def test_mixed_shared_shape(fun: Callable[..., tf.Tensor]) -> None:
    inducing_variable = mf.SharedIndependentInducingVariables(make_ip())
    kernel = mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W)
    if fun is mo_kuus.Kuu:
        t = tf.linalg.cholesky(fun(inducing_variable, kernel, jitter=1e-9))
        assert t.shape == (2, 10, 10)
    else:
        t = fun(inducing_variable, kernel, Datum.Xnew)
        assert t.shape == (2, 10, 100)
