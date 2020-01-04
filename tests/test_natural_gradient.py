from typing import Optional
import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.config import default_float
from gpflow.optimizers import NaturalGradient
from gpflow.utilities import set_trainable


class Setup:
    N, M, D = 4, 3, 2
    likelihood_variance = 0.1


@pytest.fixture
def data():
    N, D = Setup.N, Setup.D
    x = tf.random.normal((N, D), dtype=default_float())
    y = tf.random.normal((N, 1), dtype=default_float())
    return (x, y)


@pytest.fixture
def inducing_variable():
    return tf.random.normal((Setup.M, Setup.D), dtype=default_float())


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
    vgp.q_mu.trainable = True
    vgp.q_sqrt.trainable = True
    return gpr, vgp


@pytest.fixture
def sgpr_and_svgp(data, inducing_variable, kernel, likelihood):
    svgp = gpflow.models.SVGP(kernel, likelihood, inducing_variable)
    sgpr = gpflow.models.SGPR(data, kernel, inducing_variable=inducing_variable)
    sgpr.likelihood.variance.assign(Setup.likelihood_variance)
    set_trainable(svgp, False)
    svgp.q_mu.trainable = True
    svgp.q_sqrt.trainable = True
    return sgpr, svgp


def assert_gpr_vs_vgp(m1: tf.Module,
                      m2: tf.Module,
                      gamma: float = 1.0,
                      maxiter: int = 1,
                      xi_transform: Optional[gpflow.optimizers.natgrad.XiTransform] = None):
    assert maxiter >= 1

    m2_ll_before = m2.log_likelihood()
    m1_ll_before = m1.log_likelihood()

    assert m2_ll_before != m1_ll_before

    @tf.function(autograph=False)
    def loss_cb() -> tf.Tensor:
        return - m2.log_marginal_likelihood()

    params = (m2.q_mu, m2.q_sqrt)
    if xi_transform is not None:
        params += (xi_transform, )

    opt = NaturalGradient(gamma)

    @tf.function(autograph=False)
    def minimize_step():
        opt.minimize(loss_cb, var_list=[params])

    for _ in range(maxiter):
        minimize_step()

    m2_ll_after = m2.log_likelihood()
    m1_ll_after = m1.log_likelihood()

    np.testing.assert_allclose(m1_ll_after, m2_ll_after, atol=1e-4)


def assert_sgpr_vs_svgp(m1: tf.Module, m2: tf.Module):
    data = m1.data

    m1_ll_before = m1.log_likelihood()
    m2_ll_before = m2.log_likelihood(data)

    assert m2_ll_before != m1_ll_before

    @tf.function(autograph=False)
    def loss_cb() -> tf.Tensor:
        return - m2.log_marginal_likelihood(data)

    params = [(m2.q_mu, m2.q_sqrt)]
    opt = NaturalGradient(1.)
    opt.minimize(loss_cb, var_list=params)

    m1_ll_after = m1.log_likelihood()
    m2_ll_after = m2.log_likelihood(data)

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
    def meanvarsqrt_to_xi(self, mean: tf.Tensor, varsqrt: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.meanvarsqrt_to_expectation(mean, varsqrt)

    def xi_to_meanvarsqrt(self, xi1: tf.Tensor, xi2: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.expectation_to_meanvarsqrt(xi1, xi2)

    def naturals_to_xi(self, nat1: tf.Tensor, nat2: tf.Tensor) -> tf.Tensor:
        return gpflow.optimizers.natgrad.natural_to_expectation(nat1, nat2)


@pytest.mark.parametrize("xi_transform", [gpflow.optimizers.XiSqrtMeanVar(), XiEta()])
def test_xi_transform_vgp_vs_gpr(gpr_and_vgp, xi_transform):
    """
    With other transforms the solution is not given in a single step, but it should still give the same answer
    after a number of smaller steps.
    """
    gpr, vgp = gpr_and_vgp
    assert_gpr_vs_vgp(gpr, vgp, gamma=0.01, xi_transform=xi_transform, maxiter=500)
