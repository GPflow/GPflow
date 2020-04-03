import pytest
import gpflow


@pytest.mark.parametrize("config_lower", [0.0, 1e-6])
@pytest.mark.parametrize("init_lower", [None, 0.0, 1e-6])
def test_gaussian_lower_bound_constructor_check(config_lower, init_lower):
    with gpflow.config.as_context():
        gpflow.config.set_default_positive_minimum(config_lower)
        with pytest.raises(
            ValueError, match="variance of the Gaussian likelihood must be strictly greater than"
        ):
            _ = gpflow.likelihoods.Gaussian(variance=0.0, variance_lower_bound=init_lower)
