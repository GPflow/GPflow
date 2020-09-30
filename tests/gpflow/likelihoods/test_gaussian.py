import pytest

import gpflow


@pytest.mark.parametrize(
    "init_lower", [0.0, 1e-6, gpflow.likelihoods.Gaussian.DEFAULT_VARIANCE_LOWER_BOUND]
)
def test_gaussian_lower_bound_constructor_check(init_lower):
    with pytest.raises(
        ValueError, match="variance of the Gaussian likelihood must be strictly greater than"
    ):
        _ = gpflow.likelihoods.Gaussian(variance=0.0, variance_lower_bound=init_lower)
