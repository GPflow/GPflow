from typing import Tuple

import numpy as np
import pytest
from check_shapes import check_shapes

import gpflow
from gpflow.base import AnyNDArray
from gpflow.models.gpr import GPR_deprecated, GPR_with_posterior
from gpflow.posteriors import PrecomputeCacheType


@check_shapes(
    "regression_data[0]: [N, D]",
    "regression_data[1]: [N, P]",
)
def make_models(
    regression_data: gpflow.base.RegressionData,
) -> Tuple[GPR_deprecated, GPR_with_posterior]:
    """Helper function to create models"""

    k = gpflow.kernels.Matern52()

    mold = GPR_deprecated(data=regression_data, kernel=k)
    mnew = GPR_with_posterior(data=regression_data, kernel=k)
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



@pytest.mark.parametrize("cache_type", [PrecomputeCacheType.TENSOR, PrecomputeCacheType.VARIABLE])
@pytest.mark.parametrize("full_cov", [False])
@pytest.mark.parametrize("full_output_cov", [False])
def test_predict_y_vs_predict_y_faster(
    cache_type: PrecomputeCacheType, full_cov: bool, full_output_cov: bool
) -> None:
    X, X_new, Y = _get_data_for_tests()
    mold, mnew = make_models((X, Y))

    mu_old, var2_old = mold.predict_y(X_new, full_cov=full_cov, full_output_cov=full_output_cov)
    mu_new_cache, var2_new_cache = mnew.predict_y_faster(
        X_new, posteriors=mnew.posterior(cache_type), full_cov=full_cov, full_output_cov=full_output_cov
    )

    # check new cache is same as old version
    np.testing.assert_allclose(mu_old, mu_new_cache)
    np.testing.assert_allclose(var2_old, var2_new_cache)
