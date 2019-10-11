import gpflow
import pytest


def test_log_prior_with_no_prior():
    """
    A parameter without any prior should have zero log-prior,
    even if it has a transform to constrain it.
    """
    param = gpflow.Parameter(5.3, transform=gpflow.positive())
    assert param.log_prior().numpy() == 0.0
