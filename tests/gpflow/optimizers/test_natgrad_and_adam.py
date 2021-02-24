from typing import Optional

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow import set_trainable
from gpflow.optimizers import JointNaturalGradientAndAdam

from .test_natural_gradient import (
    Setup,
    assert_different,
    data,
    inducing_variable,
    kernel,
    likelihood,
)


@pytest.fixture
def gpr_and_vgp_fixed_nonvariational(data, kernel, likelihood):
    vgp = gpflow.models.VGP(data, kernel, likelihood)
    gpr = gpflow.models.GPR(data, kernel)
    gpr.likelihood.variance.assign(likelihood.variance)
    set_trainable(vgp, False)
    set_trainable(vgp.q_mu, True)
    set_trainable(vgp.q_sqrt, True)
    return gpr, vgp


@pytest.fixture
def gpr_and_vgp(data, kernel, likelihood):
    vgp = gpflow.models.VGP(data, kernel, likelihood)
    gpr = gpflow.models.GPR(data, kernel)
    gpr.likelihood.variance.assign(likelihood.variance)
    set_trainable(vgp.q_mu, False)
    set_trainable(vgp.q_sqrt, False)
    return gpr, vgp


@pytest.fixture
def sgpr_and_svgp_fixed_nonvariational(data, inducing_variable, kernel, likelihood):
    svgp = gpflow.models.SVGP(kernel, likelihood, inducing_variable)
    sgpr = gpflow.models.SGPR(data, kernel, inducing_variable=inducing_variable)
    sgpr.likelihood.variance.assign(Setup.likelihood_variance)
    set_trainable(svgp, False)
    set_trainable(svgp.q_mu, True)
    set_trainable(svgp.q_sqrt, True)
    return sgpr, svgp


@pytest.fixture
def sgpr_and_svgp(data, inducing_variable, kernel, likelihood):
    svgp = gpflow.models.SVGP(kernel, likelihood, inducing_variable)
    sgpr = gpflow.models.SGPR(data, kernel, inducing_variable=inducing_variable)
    sgpr.likelihood.variance.assign(Setup.likelihood_variance)
    set_trainable(svgp.q_mu, False)
    set_trainable(svgp.q_sqrt, False)
    return sgpr, svgp


def assert_gpr_vs_vgp_fixed_nonvariational(
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

    opt = JointNaturalGradientAndAdam(gamma, adam_lr=0.001)

    @tf.function
    def minimize_step():
        opt.minimize(
            m2.training_loss, variational_var_list=[params], non_variational_var_list=[],
        )

    for _ in range(maxiter):
        minimize_step()

    m1_ll_after = m1.training_loss()
    m2_ll_after = m2.training_loss()

    np.testing.assert_allclose(m1_ll_after, m2_ll_after, atol=1e-4)


def assert_gpr_vs_vgp(
    m1: gpflow.models.BayesianModel,
    m2: gpflow.models.BayesianModel,
    gamma: float = 1.0,
    maxiter: int = 10,
):
    assert maxiter >= 1

    m1_ll_before = m1.training_loss()
    m2_ll_before = m2.training_loss()
    m2_kernel_before = m2.kernel.lengthscales.numpy()

    assert_different(m2_ll_before, m1_ll_before)

    params = [(m2.q_mu, m2.q_sqrt)]

    opt = JointNaturalGradientAndAdam(gamma, adam_lr=0.1)

    @tf.function
    def minimize_step():
        opt.minimize(
            m2.training_loss,
            variational_var_list=params,
            non_variational_var_list=m2.trainable_variables,
        )

    for _ in range(maxiter):
        minimize_step()

    m2_ll_after = m2.training_loss()
    m2_kernel_after = m2.kernel.lengthscales.numpy()

    # Check that the parameters before/after optimization have changed (due to Adam)
    assert_different(m2_kernel_before, m2_kernel_after)

    # Check that the loss has decreased
    assert m2_ll_after < m2_ll_before


def assert_sgpr_vs_svgp_fixed_nonvariational(
    m1: gpflow.models.BayesianModel, m2: gpflow.models.BayesianModel,
):
    data = m1.data

    m1_ll_before = m1.training_loss()
    m2_ll_before = m2.training_loss(data)

    assert_different(m2_ll_before, m1_ll_before)

    params = [(m2.q_mu, m2.q_sqrt)]

    opt = JointNaturalGradientAndAdam(1.0, adam_lr=0.001)
    opt.minimize(
        m2.training_loss_closure(data),
        variational_var_list=params,
        non_variational_var_list=m2.trainable_variables,
    )

    m1_ll_after = m1.training_loss()
    m2_ll_after = m2.training_loss(data)

    np.testing.assert_allclose(m1_ll_after, m2_ll_after, atol=1e-4)


def assert_sgpr_vs_svgp(
    m1: gpflow.models.BayesianModel, m2: gpflow.models.BayesianModel, maxiter: int = 10
):
    data = m1.data

    m1_ll_before = m1.training_loss()
    m2_ll_before = m2.training_loss(data)
    m2_kernel_before = m2.kernel.lengthscales.numpy()

    assert_different(m2_ll_before, m1_ll_before)

    params = [(m2.q_mu, m2.q_sqrt)]

    opt = JointNaturalGradientAndAdam(1.0, adam_lr=1.0)

    @tf.function
    def minimize_step():
        opt.minimize(
            m2.training_loss_closure(data),
            variational_var_list=params,
            non_variational_var_list=m2.trainable_variables,
        )

    for _ in range(maxiter):
        minimize_step()

    m2_ll_after = m2.training_loss(data)
    m2_kernel_after = m2.kernel.lengthscales.numpy()

    # Check that the parameters before/after optimization have changed (due to Adam)
    assert_different(m2_kernel_before, m2_kernel_after)

    # Check that the loss has decreased
    assert m2_ll_after < m2_ll_before


def test_vgp_vs_gpr_fixed_nonvariational(gpr_and_vgp_fixed_nonvariational):
    """
    With a Gaussian likelihood Gaussian variational (VGP) model should be equivalent to the exact
    regression model (GPR) after a single nat grad+Adam step of size 1. This is
    because the non-variational parameters are fixed, so only a NatGrad step of size 1 is performed on the
    variational parameters.
    """
    gpr, vgp = gpr_and_vgp_fixed_nonvariational
    assert_gpr_vs_vgp_fixed_nonvariational(gpr, vgp)


def test_vgp_vs_gpr(gpr_and_vgp):
    """
    Tests a couple of things:
    1) Due to ADAM step, the VGP non-variational parameters must have moved
    2) The loss must have decreased due to optimization step
    """
    gpr, vgp = gpr_and_vgp
    assert_gpr_vs_vgp(gpr, vgp)


def test_svgp_vs_sgpr_fixed_nonvariational(sgpr_and_svgp_fixed_nonvariational):
    """
    With a Gaussian likelihood the sparse Gaussian variational (SVGP) model
    should be equivalent to the analytically optimal sparse regression model (SGPR)
    after a single nat grad step of size 1.0. This is
    because the non-variational parameters are fixed, so only a NatGrad step of size 1 is performed on the
    variational parameters.
    """
    sgpr, svgp = sgpr_and_svgp_fixed_nonvariational
    assert_sgpr_vs_svgp_fixed_nonvariational(sgpr, svgp)


def test_svgp_vs_sgpr(sgpr_and_svgp):
    """
    Tests a couple of things:
    1) Due to ADAM step, the SVGP non-variational parameters must have moved
    2) The loss must have decreased due to optimization step
    """
    sgpr, svgp = sgpr_and_svgp
    assert_sgpr_vs_svgp(sgpr, svgp)


def test_config():
    """
    Test config functionality of JointNaturalGradientAndAdam
    """
    gamma, adam_lr = 0.5, 0.01
    opt = JointNaturalGradientAndAdam(gamma, adam_lr=adam_lr)
    opt_config = opt.get_config()

    assert opt_config["gamma"] == gamma
    assert opt_config["adam_lr"] == adam_lr
