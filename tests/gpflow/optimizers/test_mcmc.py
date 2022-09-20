from typing import Tuple

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.distributions import Gamma, Uniform

import gpflow
from gpflow.base import AnyNDArray, PriorOn
from gpflow.experimental.check_shapes import ShapeChecker, check_shapes
from gpflow.models import GPR
from gpflow.utilities import to_default_float

np.random.seed(1)


def build_data() -> Tuple[AnyNDArray, AnyNDArray]:
    cs = ShapeChecker().check_shape
    N = 30
    X = cs(np.random.rand(N, 1), "[N, 1]")
    Y = cs(np.sin(12 * X) + 0.66 * np.cos(25 * X) + np.random.randn(N, 1) * 0.1 + 3, "[N, 1]")
    return (X, Y)


@check_shapes(
    "data[0]: [N, 1]",
    "data[1]: [N, 1]",
)
def build_model(data: Tuple[AnyNDArray, AnyNDArray]) -> GPR:

    kernel = gpflow.kernels.Matern52(lengthscales=0.3)

    meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    model = GPR(data, kernel, meanf, noise_variance=0.01)

    for p in model.parameters:
        p.prior = Gamma(to_default_float(1.0), to_default_float(1.0))

    return model


@check_shapes(
    "data[0]: [N, 1]",
    "data[1]: [N, 1]",
)
def build_model_with_uniform_prior_no_transforms(
    data: Tuple[AnyNDArray, AnyNDArray], prior_on: PriorOn, prior_width: float
) -> GPR:
    def parameter(value: tf.Tensor) -> gpflow.Parameter:
        low_value = -100
        high_value = low_value + prior_width
        prior = Uniform(low=np.float64(low_value), high=np.float64(high_value))
        return gpflow.Parameter(
            value, transform=tfp.bijectors.Identity(), prior=prior, prior_on=prior_on
        )

    k = gpflow.kernels.Matern52(lengthscales=0.3)
    k.variance = parameter(k.variance)
    k.lengthscales = parameter(k.lengthscales)

    mf = gpflow.mean_functions.Linear(1.0, 0.0)
    mf.A = parameter(mf.A)
    mf.b = parameter(mf.b)
    m = GPR(data, k, mf, noise_variance=0.01)
    m.likelihood.variance = parameter(m.likelihood.variance)
    return m


def test_mcmc_helper_parameters() -> None:
    data = build_data()
    model = build_model(data)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    for i in range(len(model.trainable_parameters)):
        assert model.trainable_parameters[i].shape == hmc_helper.current_state[i].shape
        assert model.trainable_parameters[i] == hmc_helper._parameters[i]
        assert model.trainable_parameters[i].unconstrained_variable == hmc_helper.current_state[i]


def test_mcmc_helper_target_function_constrained() -> None:
    """Set up priors on the model parameters such that we can
    readily compute their expected values."""
    config = gpflow.config.Config(positive_bijector="exp")
    with gpflow.config.as_context(config):
        data = build_data()
        model = build_model(data)

    prior_width = 200.0

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )
    target_log_prob_fn = hmc_helper.target_log_prob_fn

    # Priors which are set on the constrained space
    expected_log_prior = 0.0
    for param in model.trainable_parameters:
        if param.numpy() < 1e-3:
            # Avoid values which would be pathological for the Exp transform
            param.assign(1.0)

        low_value = -100
        high_value = low_value + prior_width

        param.prior = Uniform(low=np.float64(low_value), high=np.float64(high_value))
        param.prior_on = PriorOn.CONSTRAINED

        prior_density_on_constrained = 1 / prior_width
        prior_density_on_unconstrained = prior_density_on_constrained * param.numpy()

        expected_log_prior += np.log(prior_density_on_unconstrained)

    log_marginal_likelihood = model.log_marginal_likelihood().numpy()
    expected_log_prob = log_marginal_likelihood + expected_log_prior

    np.testing.assert_allclose(target_log_prob_fn(), expected_log_prob, rtol=1e-6)


def test_mcmc_helper_target_function_unconstrained() -> None:
    """
    Verifies the objective for a set of priors which are defined on the unconstrained space.
    """
    # Set up priors on the model parameters such that we can readily compute their expected values.
    expected_log_prior = 0.0
    prior_width = 200.0

    data = build_data()
    model = build_model_with_uniform_prior_no_transforms(data, PriorOn.UNCONSTRAINED, prior_width)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    for _ in model.trainable_parameters:
        prior_density = 1 / prior_width
        expected_log_prior += np.log(prior_density)

    target_log_prob_fn = hmc_helper.target_log_prob_fn
    expected_log_prob = model.log_marginal_likelihood().numpy() + expected_log_prior

    np.testing.assert_allclose(target_log_prob_fn(), expected_log_prob)


@pytest.mark.parametrize("prior_on", [PriorOn.CONSTRAINED, PriorOn.UNCONSTRAINED])
def test_mcmc_helper_target_function_no_transforms(prior_on: PriorOn) -> None:
    """Verifies the objective for a set of priors where no transforms are set."""
    expected_log_prior = 0.0
    prior_width = 200.0

    data = build_data()
    model = build_model_with_uniform_prior_no_transforms(data, prior_on, prior_width)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    for _ in model.trainable_parameters:
        prior_density = 1 / prior_width
        expected_log_prior += np.log(prior_density)

    log_marginal_likelihood = model.log_marginal_likelihood().numpy()
    expected_log_prob = log_marginal_likelihood + expected_log_prior
    target_log_prob_fn = hmc_helper.target_log_prob_fn

    np.testing.assert_allclose(target_log_prob_fn(), expected_log_prob)

    # Test the wrapped closure
    log_prob, grad_fn = target_log_prob_fn.__original_wrapped__()  # type: ignore[attr-defined]
    grad, nones = grad_fn(1, [None] * len(model.trainable_parameters))
    assert len(grad) == len(model.trainable_parameters)
    assert nones == [None] * len(model.trainable_parameters)


def test_mcmc_sampler_integration() -> None:
    data = build_data()
    model = build_model(data)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.log_posterior_density, model.trainable_parameters
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=0.01,
    )

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=2,
        target_accept_prob=gpflow.utilities.to_default_float(0.75),
        adaptation_rate=0.1,
    )

    num_samples = 5

    @tf.function
    def run_chain_fn() -> tfp.mcmc.StatesAndTrace:
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=2,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted,
        )

    samples, _ = run_chain_fn()

    assert len(samples) == len(model.trainable_parameters)
    parameter_samples = hmc_helper.convert_to_constrained_values(samples)
    assert len(parameter_samples) == len(samples)

    for i in range(len(model.trainable_parameters)):
        assert len(samples[i]) == num_samples
        assert hmc_helper.current_state[i].numpy() == samples[i][-1]
        assert hmc_helper._parameters[i].numpy() == parameter_samples[i][-1]


def test_helper_with_variables_fails() -> None:
    variable = tf.Variable(0.1)
    with pytest.raises(
        ValueError, match=r"`parameters` should only contain gpflow.Parameter objects with priors"
    ):
        gpflow.optimizers.SamplingHelper(lambda: variable ** 2, (variable,))
