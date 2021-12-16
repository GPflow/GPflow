import numpy as np
import pytest

import gpflow
from gpflow.models.gpr import GPR_deprecated, GPR_with_posterior
from gpflow.posteriors import PrecomputeCacheType


def make_models(regression_data):
    """Helper function to create models"""

    k = gpflow.kernels.Matern52()

    mold = GPR_deprecated(data=regression_data, kernel=k)
    mnew = GPR_with_posterior(data=regression_data, kernel=k)
    return mold, mnew


def _get_data_for_tests():
    """Helper function to create testing data"""
    X = np.random.randn(5, 6)
    Y = np.random.randn(5, 2)
    X_new = np.random.randn(3, 10, 5, 6)
    return X, X_new, Y


@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_old_vs_new_gp_fused(
    cache_type: PrecomputeCacheType, full_cov: bool, full_output_cov: bool
):
    X, X_new, Y = _get_data_for_tests()
    mold, mnew = make_models((X, Y))

    mu_old, var2_old = mold.predict_f(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_new_fuse, var2_new_fuse = mnew.predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )
    # check new fuse is same as old version
    np.testing.assert_allclose(mu_new_fuse, mu_old)
    np.testing.assert_allclose(var2_new_fuse, var2_old)


@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_old_vs_new_with_posterior(
    cache_type: PrecomputeCacheType, full_cov: bool, full_output_cov: bool
):
    X, X_new, Y = _get_data_for_tests()
    mold, mnew = make_models((X, Y))

    mu_old, var2_old = mold.predict_f(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_new_cache, var2_new_cache = mnew.posterior(cache_type).predict_f(
        X_new, full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_old, mu_new_cache)
    np.testing.assert_allclose(var2_old, var2_new_cache)
