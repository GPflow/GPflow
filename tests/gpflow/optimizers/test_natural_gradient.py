from typing import Optional

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow import set_trainable
from gpflow.config import default_float
from gpflow.optimizers import NaturalGradient


class Setup:
    N, M, D = 4, 3, 2
    likelihood_variance = 0.1
    rng = np.random.RandomState(42)
    X = rng.randn(N, D)
    Y = rng.randn(N, 1)
    Z = rng.randn(M, D)


@pytest.fixture
def data():
    X = tf.convert_to_tensor(Setup.X, dtype=default_float())
    Y = tf.convert_to_tensor(Setup.Y, dtype=default_float())
    return (X, Y)


@pytest.fixture
def inducing_variable():
    Z = tf.convert_to_tensor(Setup.Z, dtype=default_float())
    return Z


@pytest.fixture
def kernel():
    return gpflow.kernels.SquaredExponential()


@pytest.fixture
def likelihood():
    return gpflow.likelihoods.Gaussian(variance=Setup.likelihood_variance)


@pytest.fixture
def gpr_and_vgp(data, kernel, likelihood):
    vgp = gpflow.models.VGP(data, kernel, likelihood)
    gpr = gpflow.models.GPR(data, kernel)
    gpr.likelihood.variance.assign(likelihood.variance)
    set_trainable(vgp, False)
    set_trainable(vgp.q_mu, True)
    set_trainable(vgp.q_sqrt, True)
    return gpr, vgp


@pytest.fixture
def sgpr_and_svgp(data, inducing_variable, kernel, likelihood):
    svgp = gpflow.models.SVGP(kernel, likelihood, inducing_variable)
    sgpr = gpflow.models.SGPR(data, kernel, inducing_variable=inducing_variable)
    sgpr.likelihood.variance.assign(Setup.likelihood_variance)
    set_trainable(svgp, False)
    set_trainable(svgp.q_mu, True)
    set_trainable(svgp.q_sqrt, True)
    return sgpr, svgp


def assert_different(value1, value2, rtol=0.07):
    """ assert relative difference > rtol """
    relative_difference = (value1 - value2) / (value1 + value2)
    assert np.abs(relative_difference) > rtol


def assert_gpr_vs_vgp(
    m1: gpflow.models.BayesianModel,
    m2: gpflow.models.BayesianModel,
    gamma: float = 1.0,
    maxiter: int = 1,
    xi_transform: Optional[gpflow.optimizers.natgrad.XiTransform] = None,
):
    assert maxiter >= 1

    m1_ll_before = m1.training_loss()
    m2_ll_before = m2.training_loss()

    assert_different(m2_ll_before, m1_ll_before)

    params = (m2.q_mu, m2.q_sqrt)
    if xi_transform is not None:
        params += (xi_transform,)

    opt = NaturalGradient(gamma)

    @tf.function
    def minimize_step():
        opt.minimize(m2.training_loss, var_list=[params])

    for _ in range(maxiter):
        minimize_step()

    m1_ll_after = m1.training_loss()
    m2_ll_after = m2.training_loss()

    np.testing.assert_allclose(m1_ll_after, m2_ll_after, atol=1e-4)


def assert_sgpr_vs_svgp(
    m1: gpflow.models.BayesianModel, m2: gpflow.models.BayesianModel,
):
    data = m1.data

    m1_ll_before = m1.training_loss()
    m2_ll_before = m2.training_loss(data)

    assert_different(m2_ll_before, m1_ll_before)

    params = [(m2.q_mu, m2.q_sqrt)]
    opt = NaturalGradient(1.0)
    opt.minimize(m2.training_loss_closure(data), var_list=params)

    m1_ll_after = m1.training_loss()
    m2_ll_after = m2.training_loss(data)

    np.testing.assert_allclose(m1_ll_after, m2_ll_after, atol=1e-4)


def test_vgp_vs_gpr(gpr_and_vgp):
    """
    With a Gaussian likelihood the Gaussian variational (VGP) model should be equivalent to the exact
    regression model (GPR) after a single nat grad step of size 1
    """
    gpr, vgp = gpr_and_vgp
    assert_gpr_vs_vgp(gpr, vgp)


def test_small_q_sqrt_handeled_correctly(gpr_and_vgp, data):
    """
    This is an extra test to make sure things still work when q_sqrt is small. This was breaking (#767)
    """
    gpr, vgp = gpr_and_vgp
    vgp.q_mu.assign(np.random.randn(data[0].shape[0], 1))
    vgp.q_sqrt.assign(np.eye(data[0].shape[0])[None, :, :] * 1e-3)
    assert_gpr_vs_vgp(gpr, vgp)


def test_svgp_vs_sgpr(sgpr_and_svgp):
    """
    With a Gaussian likelihood the sparse Gaussian variational (SVGP) model
    should be equivalent to the analytically optimial sparse regression model (SGPR)
    after a single nat grad step of size 1.0
    """
    sgpr, svgp = sgpr_and_svgp
    assert_sgpr_vs_svgp(sgpr, svgp)


class XiEta(gpflow.optimizers.XiTransform):
    @staticmethod
    def meanvarsqrt_to_xi(mean: tf.Tensor, varsqrt: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.meanvarsqrt_to_expectation(mean, varsqrt)

    @staticmethod
    def xi_to_meanvarsqrt(xi1: tf.Tensor, xi2: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.expectation_to_meanvarsqrt(xi1, xi2)

    @staticmethod
    def naturals_to_xi(nat1: tf.Tensor, nat2: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.natural_to_expectation(nat1, nat2)


@pytest.mark.parametrize("xi_transform", [gpflow.optimizers.XiSqrtMeanVar(), XiEta()])
def test_xi_transform_vgp_vs_gpr(gpr_and_vgp, xi_transform):
    """
    With other transforms the solution is not given in a single step, but it should still give the same answer
    after a number of smaller steps.
    """
    gpr, vgp = gpr_and_vgp
    assert_gpr_vs_vgp(gpr, vgp, gamma=0.01, xi_transform=xi_transform, maxiter=500)
