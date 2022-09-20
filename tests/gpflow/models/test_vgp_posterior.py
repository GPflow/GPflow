from typing import Tuple

import numpy as np
import pytest

import gpflow
from gpflow.base import AnyNDArray, RegressionData
from gpflow.experimental.check_shapes import check_shapes
from gpflow.models.vgp import VGP_deprecated, VGP_with_posterior
from gpflow.posteriors import PrecomputeCacheType


@check_shapes(
    "regression_data[0]: [N, D]",
    "regression_data[1]: [N, P]",
)
def make_models(
    regression_data: RegressionData, likelihood: gpflow.likelihoods.Likelihood
) -> Tuple[VGP_deprecated, VGP_with_posterior]:
    """Helper function to create models"""

    k = gpflow.kernels.Matern52()
    likelihood = gpflow.likelihoods.Gaussian()

    mold = VGP_deprecated(data=regression_data, kernel=k, likelihood=likelihood)
    mnew = VGP_with_posterior(data=regression_data, kernel=k, likelihood=likelihood)
    return mold, mnew


@check_shapes(
    "return[0]: [N, D]  # X",
    "return[1]: [batch_new..., N_new, D]  # X_new",
    "return[2]: [N, P]  # Y",
)
def _get_data_for_tests() -> Tuple[AnyNDArray, AnyNDArray, AnyNDArray]:
    """Helper function to create testing data"""
    X = np.random.randn(5, 6)
    Y = np.random.randn(5, 2)
    X_new = np.random.randn(3, 10, 5, 6)
    return X, X_new, Y


@pytest.mark.parametrize(
    "likelihood", [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Exponential()]
)
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_old_vs_new_gp_fused(
    likelihood: gpflow.likelihoods.Likelihood,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    X, X_new, Y = _get_data_for_tests()
    mold, mnew = make_models((X, Y), likelihood)

    mu_old, var2_old = mold.predict_f(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_new_fuse, var2_new_fuse = mnew.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    # check new fuse is same as old version
    np.testing.assert_allclose(mu_new_fuse, mu_old)
    np.testing.assert_allclose(var2_new_fuse, var2_old)


@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
@pytest.mark.parametrize(
    "likelihood", [gpflow.likelihoods.Gaussian(), gpflow.likelihoods.Exponential()]
)
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_old_vs_new_with_posterior(
    cache_type: PrecomputeCacheType,
    likelihood: gpflow.likelihoods.Likelihood,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    X, X_new, Y = _get_data_for_tests()
    mold, mnew = make_models((X, Y), likelihood)

    mu_old, var2_old = mold.predict_f(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_new_cache, var2_new_cache = mnew.posterior(cache_type).predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_old, mu_new_cache)
    np.testing.assert_allclose(var2_old, var2_new_cache)
