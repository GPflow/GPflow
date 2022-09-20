from typing import Any, Callable, Sequence

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow import covariances
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


def type_name(x: Any) -> str:
    return x.__class__.__name__  # type: ignore[no-any-return]


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
    (
        mf.SharedIndependentInducingVariables(make_ip()),
        mf.SharedIndependentInducingVariables(make_ip()),
    ),
    (
        mf.SeparateIndependentInducingVariables(make_ips(Datum.P)),
        mf.SeparateIndependentInducingVariables(make_ips(Datum.L)),
    ),
]

multioutput_fallback_inducing_variable_list = [
    mf.FallbackSharedIndependentInducingVariables(make_ip()),
    mf.FallbackSeparateIndependentInducingVariables(make_ips(Datum.L)),
]

multioutput_kernel_list = [
    mk.SharedIndependent(make_kernel(), Datum.P),
    mk.SeparateIndependent(make_kernels(Datum.P)),
    mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W),
]


@pytest.mark.parametrize(
    "inducing_variable_P,inducing_variable_L", multioutput_inducing_variable_list, ids=type_name
)
@pytest.mark.parametrize("kernel", multioutput_kernel_list, ids=type_name)
def test_kuu_shape(
    inducing_variable_P: InducingVariables, inducing_variable_L: InducingVariables, kernel: Kernel
) -> None:
    if isinstance(kernel, mk.LinearCoregionalization):
        inducing_variable = inducing_variable_L
    else:
        inducing_variable = inducing_variable_P

    Kuu = covariances.Kuu(inducing_variable, kernel, jitter=1e-9)
    t = tf.linalg.cholesky(Kuu)

    if isinstance(kernel, mk.SharedIndependent):
        if isinstance(inducing_variable, mf.SeparateIndependentInducingVariables):
            assert t.shape == (Datum.P, Datum.M, Datum.M)
        else:
            assert t.shape == (Datum.M, Datum.M)
    elif isinstance(kernel, mk.LinearCoregionalization):
        assert t.shape == (Datum.L, Datum.M, Datum.M)
    else:
        assert t.shape == (Datum.P, Datum.M, Datum.M)


@pytest.mark.parametrize(
    "inducing_variable_P,inducing_variable_L", multioutput_inducing_variable_list, ids=type_name
)
@pytest.mark.parametrize("kernel", multioutput_kernel_list, ids=type_name)
def test_kuf_shape(
    inducing_variable_P: InducingVariables, inducing_variable_L: InducingVariables, kernel: Kernel
) -> None:
    if isinstance(kernel, mk.LinearCoregionalization):
        inducing_variable = inducing_variable_L
    else:
        inducing_variable = inducing_variable_P

    Kuf = covariances.Kuf(inducing_variable, kernel, Datum.Xnew)

    if isinstance(kernel, mk.SharedIndependent):
        if isinstance(inducing_variable, mf.SeparateIndependentInducingVariables):
            assert Kuf.shape == (Datum.P, Datum.M, Datum.N)
        else:
            assert Kuf.shape == (Datum.M, Datum.N)
    elif isinstance(kernel, mk.LinearCoregionalization):
        assert Kuf.shape == (Datum.L, Datum.M, Datum.N)
    else:
        assert Kuf.shape == (Datum.P, Datum.M, Datum.N)


@pytest.mark.parametrize(
    "inducing_variable", multioutput_fallback_inducing_variable_list, ids=type_name
)
def test_kuf_fallback_shared_inducing_variables_shape(inducing_variable: InducingVariables) -> None:
    kernel = mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W)
    Kuf = covariances.Kuf(inducing_variable, kernel, Datum.Xnew)

    assert Kuf.shape == (Datum.M, Datum.L, Datum.N, Datum.P)


@pytest.mark.parametrize("fun", [covariances.Kuu, covariances.Kuf], ids=type_name)
def test_mixed_shared_shape(fun: Callable[..., tf.Tensor]) -> None:
    inducing_variable = mf.SharedIndependentInducingVariables(make_ip())
    kernel = mk.LinearCoregionalization(make_kernels(Datum.L), Datum.W)
    if fun is covariances.Kuu:
        t = tf.linalg.cholesky(fun(inducing_variable, kernel, jitter=1e-9))
        assert t.shape == (Datum.L, Datum.M, Datum.M)
    else:
        t = fun(inducing_variable, kernel, Datum.Xnew)
        assert t.shape == (Datum.L, Datum.M, Datum.N)
