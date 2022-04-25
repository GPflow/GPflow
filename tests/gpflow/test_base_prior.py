from typing import Any, Tuple, Type

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.bijectors import Exp
from tensorflow_probability.python.distributions import Uniform

import gpflow
from gpflow.base import AnyNDArray, PriorOn
from gpflow.config import set_default_float
from gpflow.utilities import to_default_float

np.random.seed(1)


class Datum:
    X: AnyNDArray = 10 * np.random.randn(5, 1)
    Y: AnyNDArray = 10 * np.random.randn(5, 1)
    lengthscale = 3.3


def test_gpr_objective_equivalence() -> None:
    """
    In Maximum Likelihood Estimation (MLE), i.e. when there are no priors on
    the parameters, the objective should not depend on any transforms on the
    parameters.
    We use GPR as a simple model that has an objective.
    """
    data = (Datum.X, Datum.Y)
    l_value = Datum.lengthscale

    l_variable = tf.Variable(l_value, dtype=gpflow.default_float(), trainable=True)
    m1 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential(lengthscales=l_value))
    m2 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential())
    m2.kernel.lengthscales = gpflow.Parameter(l_variable, transform=None)
    assert np.allclose(
        m1.kernel.lengthscales.numpy(), m2.kernel.lengthscales.numpy()
    )  # consistency check

    assert np.allclose(
        m1.log_marginal_likelihood().numpy(), m2.log_marginal_likelihood().numpy()
    ), "MLE objective should not depend on Parameter transform"


def test_log_prior_with_no_prior() -> None:
    """
    A parameter without any prior should have zero log-prior,
    even if it has a transform to constrain it.
    """
    param = gpflow.Parameter(5.3, transform=gpflow.utilities.positive())
    assert param.log_prior_density().numpy() == 0.0


def test_log_prior_for_uniform_prior() -> None:
    """
    If we assign a Uniform prior to a parameter, we should not expect the value of the prior density
    to change with the parameter value, even if it has a transform associated with it.
    """

    uniform_prior = Uniform(low=np.float64(0), high=np.float64(100))
    param = gpflow.Parameter(1.0, transform=gpflow.utilities.positive(), prior=uniform_prior)
    low_value = param.log_prior_density().numpy()
    param.assign(10.0)
    high_value = param.log_prior_density().numpy()

    assert np.isclose(low_value, high_value)


def test_log_prior_on_unconstrained() -> None:
    """
    A parameter with an Exp transform, and a uniform prior on its unconstrained, should have a
    prior in the constrained space that scales as 1/value.
    """

    initial_value = 1.0
    scale_factor = 10.0
    uniform_prior = Uniform(low=np.float64(0), high=np.float64(100))
    param = gpflow.Parameter(
        initial_value,
        transform=Exp(),
        prior=uniform_prior,
        prior_on=PriorOn.UNCONSTRAINED,
    )
    low_value = param.log_prior_density().numpy()
    param.assign(scale_factor * initial_value)
    high_value = param.log_prior_density().numpy()

    assert np.isclose(low_value, high_value + np.log(scale_factor))


class DummyModel(gpflow.models.BayesianModel):
    value = 3.3
    log_scale = 0.4

    def __init__(self, with_transform: bool) -> None:
        super().__init__()

        prior = tfp.distributions.Normal(to_default_float(1.0), to_default_float(1.0))

        scale = np.exp(self.log_scale)
        if with_transform:
            transform = tfp.bijectors.Shift(to_default_float(0.0))(
                tfp.bijectors.Scale(to_default_float(scale))
            )
        else:
            transform = None

        self.theta = gpflow.Parameter(self.value, prior=prior, transform=transform)

    def maximum_log_likelihood_objective(self, *args: Any, **kwargs: Any) -> tf.Tensor:
        assert not args
        assert not kwargs
        return (self.theta + 5) ** 2


def test_map_invariance_to_transform() -> None:
    m1 = DummyModel(with_transform=True)
    m2 = DummyModel(with_transform=False)
    assert np.allclose(
        m1.log_posterior_density().numpy(), m2.log_posterior_density().numpy()
    ), "log posterior density should not be affected by a transform"


def get_gpmc_model_params() -> Tuple[Any, ...]:
    kernel = gpflow.kernels.Matern32()
    likelihood = gpflow.likelihoods.Gaussian()
    data = [np.random.randn(5, 1), np.random.randn(5, 1)]
    return data, kernel, likelihood


@pytest.mark.parametrize(
    "model_class, args",
    [
        (gpflow.models.GPMC, get_gpmc_model_params()),
        # (gpflow.models.SGPMC, get_SGPMC_model_params()) # Fails due to inducing_variable=None bug
    ],
)
def test_v_prior_dtypes(model_class: Type[Any], args: Tuple[Any, ...]) -> None:
    with gpflow.config.as_context():
        set_default_float(np.float32)
        m = model_class(*args)
        assert m.V.prior.dtype == np.float32
        set_default_float(np.float64)
        m = model_class(*args)
        assert m.V.prior.dtype == np.float64
