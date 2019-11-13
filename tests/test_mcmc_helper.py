import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import set_default_float
from gpflow.utilities import to_default_float

np.random.seed(1)


def build_data():
    N = 30
    X = np.random.rand(N, 1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N, 1)*0.1 + 3
    return (X, Y)


def build_model(data):

    kernel = gpflow.kernels.Matern52(lengthscale=0.3)

    meanf = gpflow.mean_functions.Linear(1.0, 0.0)
    model = gpflow.models.GPR(data, kernel, meanf)
    model.likelihood.variance.assign(0.01)
    return model


def test_mcmc_helper_parameters():
    data = build_data()
    model = build_model(data)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.trainable_parameters, model.log_marginal_likelihood
    )

    for i in range(len(model.trainable_parameters)):
        assert model.trainable_parameters[i].shape == hmc_helper.current_state[i].shape
        assert model.trainable_parameters[i] == hmc_helper._parameters[i]
        if isinstance(model.trainable_parameters[i], gpflow.Parameter):
            assert model.trainable_parameters[i].unconstrained_variable == hmc_helper.current_state[i]


def test_mcmc_helper_target_function():
    data = build_data()
    model = build_model(data)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.trainable_parameters, model.log_marginal_likelihood
    )

    target_log_prob_fn = hmc_helper.target_log_prob_fn

    assert model.log_marginal_likelihood() == target_log_prob_fn()

    model.likelihood.variance.assign(1)

    assert model.log_marginal_likelihood() == target_log_prob_fn()

    # test the wrapped closure
    log_prob, grad_fn = target_log_prob_fn.__original_wrapped__()
    grad, nones =  grad_fn(1, [None] * len(model.trainable_parameters))
    assert len(grad) == len(model.trainable_parameters)
    assert nones == [None] * len(model.trainable_parameters)

def test_mcmc_sampler_integration():
    data = build_data()
    model = build_model(data)

    hmc_helper = gpflow.optimizers.SamplingHelper(
        model.trainable_parameters, model.log_marginal_likelihood
    )

    hmc = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=hmc_helper.target_log_prob_fn,
        num_leapfrog_steps=2,
        step_size=0.01
    )

    adaptive_hmc = tfp.mcmc.SimpleStepSizeAdaptation(
        hmc,
        num_adaptation_steps=2,
        target_accept_prob=gpflow.utilities.to_default_float(0.75),
        adaptation_rate=0.1
    )

    num_samples = 5

    @tf.function
    def run_chain_fn():
        return tfp.mcmc.sample_chain(
            num_results=num_samples,
            num_burnin_steps=2,
            current_state=hmc_helper.current_state,
            kernel=adaptive_hmc,
            trace_fn=lambda _, pkr: pkr.inner_results.is_accepted
        )
    samples, _ = run_chain_fn()

    assert len(samples) == len(model.trainable_parameters)
    parameter_samples = hmc_helper.convert_constrained_values(samples)
    assert len(parameter_samples) == len(samples)

    for i in range(len(model.trainable_parameters)):
        assert len(samples[i]) == num_samples
        assert hmc_helper.current_state[i].numpy() == samples[i][-1]
        assert hmc_helper._parameters[i].numpy() == parameter_samples[i][-1]
