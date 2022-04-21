from itertools import product
from typing import Optional, Tuple

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.base import AnyNDArray
from gpflow.models.svgp import SVGP_deprecated, SVGP_with_posterior

input_dim = 7


def make_models(
    M: int = 64,
    D: int = input_dim,
    L: int = 3,
    q_diag: bool = False,
    whiten: bool = True,
    mo: Optional[Tuple[str, str]] = None,
) -> Tuple[SVGP_deprecated, SVGP_with_posterior]:
    if mo is None:
        k = gpflow.kernels.Matern52()
        Z = np.random.randn(M, D)
    else:
        kernel_type, iv_type = mo

        k_list = [gpflow.kernels.Matern52() for _ in range(L)]
        output_dim = 5
        w = tf.Variable(initial_value=np.random.rand(output_dim, L), dtype=tf.float64, name="w")
        if kernel_type == "LinearCoregionalization":
            k = gpflow.kernels.LinearCoregionalization(k_list, W=w)
        elif kernel_type == "SeparateIndependent":
            k = gpflow.kernels.SeparateIndependent(k_list)
        elif kernel_type == "SharedIndependent":
            k = gpflow.kernels.SharedIndependent(k_list[0], output_dim=L)
        else:
            raise NotImplementedError

        iv_list = [
            gpflow.inducing_variables.InducingPoints(np.random.randn(M, D)) for _ in range(L)
        ]
        if iv_type == "SeparateIndependent":
            Z = gpflow.inducing_variables.SeparateIndependentInducingVariables(iv_list)
        elif iv_type == "SharedIndependent":
            Z = gpflow.inducing_variables.SharedIndependentInducingVariables(iv_list[0])
        else:
            raise NotImplementedError

    lik = gpflow.likelihoods.Gaussian(0.1)
    q_mu = np.random.randn(M, L)
    if q_diag:
        q_sqrt: AnyNDArray = np.random.randn(M, L) ** 2
    else:
        q_sqrt = np.tril(np.random.randn(L, M, M))
    mold = SVGP_deprecated(k, lik, Z, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten)
    mnew = SVGP_with_posterior(k, lik, Z, q_diag=q_diag, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten)
    return mold, mnew


@pytest.mark.parametrize("q_diag", [True, False])
@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize(
    "multioutput",
    [
        None,
        *product(
            ("LinearCoregionalization", "SeparateIndependent", "SharedIndependent"),
            ("SeparateIndependent", "SharedIndependent"),
        ),
    ],
)
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_old_vs_new_svgp(
    q_diag: bool,
    white: bool,
    multioutput: Optional[Tuple[str, str]],
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    mold, mnew = make_models(q_diag=q_diag, whiten=white, mo=multioutput)

    X = np.random.randn(100, input_dim)

    mu, var = mnew.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
    mu2, var2 = mold.predict_f(X, full_cov=full_cov, full_output_cov=full_output_cov)
    np.testing.assert_allclose(mu, mu2)
    np.testing.assert_allclose(var, var2)
