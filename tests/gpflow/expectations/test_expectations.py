# Copyright 2018 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose
from tensorflow import convert_to_tensor as ctt

import gpflow
from gpflow import inducing_variables, kernels
from gpflow import mean_functions as mf
from gpflow.config import default_float
from gpflow.expectations import expectation, quadrature_expectation
from gpflow.probability_distributions import DiagonalGaussian, Gaussian, MarkovGaussian

rng = np.random.RandomState(1)
RTOL = 1e-6

num_data = 5
num_ind = 4
D_in = 2
D_out = 2

Xmu = ctt(rng.randn(num_data, D_in))
Xmu_markov = ctt(rng.randn(num_data + 1, D_in))  # (N+1)xD
Xcov = rng.randn(num_data, D_in, D_in)
Xcov = ctt(Xcov @ np.transpose(Xcov, (0, 2, 1)))
Z = rng.randn(num_ind, D_in)


def markov_gauss():
    cov_params = rng.randn(num_data + 1, D_in, 2 * D_in) / 2.0  # (N+1)xDx2D
    Xcov = cov_params @ np.transpose(cov_params, (0, 2, 1))  # (N+1)xDxD
    Xcross = cov_params[:-1] @ np.transpose(cov_params[1:], (0, 2, 1))  # NxDxD
    Xcross = np.concatenate((Xcross, np.zeros((1, D_in, D_in))), 0)  # (N+1)xDxD
    Xcov = np.stack([Xcov, Xcross])  # 2x(N+1)xDxD
    return MarkovGaussian(Xmu_markov, ctt(Xcov))


_means = {
    "lin": mf.Linear(A=rng.randn(D_in, D_out), b=rng.randn(D_out)),
    "identity": mf.Identity(input_dim=D_in),
    "const": mf.Constant(c=rng.randn(D_out)),
    "zero": mf.Zero(output_dim=D_out),
}

_distrs = {
    "gauss": Gaussian(Xmu, Xcov),
    "dirac_gauss": Gaussian(Xmu, np.zeros((num_data, D_in, D_in))),
    "gauss_diag": DiagonalGaussian(Xmu, rng.rand(num_data, D_in)),
    "dirac_diag": DiagonalGaussian(Xmu, np.zeros((num_data, D_in))),
    "dirac_markov_gauss": MarkovGaussian(Xmu_markov, np.zeros((2, num_data + 1, D_in, D_in))),
    "markov_gauss": markov_gauss(),
}

_kerns = {
    "rbf": kernels.SquaredExponential(variance=rng.rand(), lengthscales=rng.rand() + 1.0),
    "lin": kernels.Linear(variance=rng.rand()),
    "matern": kernels.Matern32(variance=rng.rand()),
    "rbf_act_dim_0": kernels.SquaredExponential(
        variance=rng.rand(), lengthscales=rng.rand() + 1.0, active_dims=[0]
    ),
    "rbf_act_dim_1": kernels.SquaredExponential(
        variance=rng.rand(), lengthscales=rng.rand() + 1.0, active_dims=[1]
    ),
    "lin_act_dim_0": kernels.Linear(variance=rng.rand(), active_dims=[0]),
    "lin_act_dim_1": kernels.Linear(variance=rng.rand(), active_dims=[1]),
    "rbf_lin_sum": kernels.Sum(
        [
            kernels.SquaredExponential(variance=rng.rand(), lengthscales=rng.rand() + 1.0),
            kernels.Linear(variance=rng.rand()),
        ]
    ),
    "rbf_lin_sum2": kernels.Sum(
        [
            kernels.Linear(variance=rng.rand()),
            kernels.SquaredExponential(variance=rng.rand(), lengthscales=rng.rand() + 1.0),
            kernels.Linear(variance=rng.rand()),
            kernels.SquaredExponential(variance=rng.rand(), lengthscales=rng.rand() + 1.0),
        ]
    ),
    "rbf_lin_prod": kernels.Product(
        [
            kernels.SquaredExponential(
                variance=rng.rand(), lengthscales=rng.rand() + 1.0, active_dims=[0]
            ),
            kernels.Linear(variance=rng.rand(), active_dims=[1]),
        ]
    ),
}


def kerns(*args):
    return [_kerns[k] for k in args]


def distrs(*args):
    return [_distrs[k] for k in args]


def means(*args):
    return [_means[k] for k in args]


@pytest.fixture
def inducing_variable():
    return inducing_variables.InducingPoints(Z)


def _check(params):
    analytic = expectation(*params)
    quad = quadrature_expectation(*params)
    assert_allclose(analytic, quad, rtol=RTOL)


# =================================== TESTS ===================================

distr_args1 = distrs("gauss")
mean_args = means("lin", "identity", "const", "zero")
kern_args1 = kerns("lin", "rbf", "rbf_lin_sum", "rbf_lin_prod")
kern_args2 = kerns("lin", "rbf", "rbf_lin_sum")


@pytest.mark.parametrize("distribution", distr_args1)
@pytest.mark.parametrize("mean1", mean_args)
@pytest.mark.parametrize("mean2", mean_args)
@pytest.mark.parametrize("arg_filter", [lambda p, m1, m2: (p, m1), lambda p, m1, m2: (p, m1, m2)])
def test_mean_function_only_expectations(distribution, mean1, mean2, arg_filter):
    params = arg_filter(distribution, mean1, mean2)
    _check(params)


@pytest.mark.parametrize("distribution", distrs("gauss", "gauss_diag"))
@pytest.mark.parametrize("kernel", kern_args1)
@pytest.mark.parametrize(
    "arg_filter",
    [lambda p, k, f: (p, k), lambda p, k, f: (p, (k, f)), lambda p, k, f: (p, (k, f), (k, f)),],
)
def test_kernel_only_expectations(distribution, kernel, inducing_variable, arg_filter):
    params = arg_filter(distribution, kernel, inducing_variable)
    _check(params)


@pytest.mark.parametrize("distribution", distr_args1)
@pytest.mark.parametrize("kernel", kerns("rbf", "lin", "matern", "rbf_lin_sum"))
@pytest.mark.parametrize("mean", mean_args)
@pytest.mark.parametrize(
    "arg_filter", [lambda p, k, f, m: (p, (k, f), m), lambda p, k, f, m: (p, m, (k, f))]
)
def test_kernel_mean_function_expectations(
    distribution, kernel, inducing_variable, mean, arg_filter
):
    params = arg_filter(distribution, kernel, inducing_variable, mean)
    _check(params)


@pytest.mark.parametrize("kernel", kern_args1)
def test_eKdiag_no_uncertainty(kernel):
    eKdiag = expectation(_distrs["dirac_diag"], kernel)
    Kdiag = kernel(Xmu, full_cov=False)
    assert_allclose(eKdiag, Kdiag, rtol=RTOL)


@pytest.mark.parametrize("kernel", kern_args1)
def test_eKxz_no_uncertainty(kernel, inducing_variable):
    eKxz = expectation(_distrs["dirac_diag"], (kernel, inducing_variable))
    Kxz = kernel(Xmu, Z)
    assert_allclose(eKxz, Kxz, rtol=RTOL)


@pytest.mark.parametrize("kernel", kern_args2)
@pytest.mark.parametrize("mean", mean_args)
def test_eMxKxz_no_uncertainty(kernel, inducing_variable, mean):
    exKxz = expectation(_distrs["dirac_diag"], mean, (kernel, inducing_variable))
    Kxz = kernel(Xmu, Z)
    xKxz = expectation(_distrs["dirac_gauss"], mean)[:, :, None] * Kxz[:, None, :]
    assert_allclose(exKxz, xKxz, rtol=RTOL)


@pytest.mark.parametrize("kernel", kern_args1)
def test_eKzxKxz_no_uncertainty(kernel, inducing_variable):
    eKzxKxz = expectation(
        _distrs["dirac_diag"], (kernel, inducing_variable), (kernel, inducing_variable)
    )
    Kxz = kernel(Xmu, Z)
    KzxKxz = Kxz[:, :, None] * Kxz[:, None, :]
    assert_allclose(eKzxKxz, KzxKxz, rtol=RTOL)


def test_RBF_eKzxKxz_gradient_notNaN():
    """
    Ensure that <K_{Z, x} K_{x, Z}>_p(x) is not NaN and correct, when
    K_{Z, Z} is zero with finite precision. See pull request #595.
    """
    kernel = gpflow.kernels.SquaredExponential(1, lengthscales=0.1)
    kernel.variance.assign(2.0)

    p = gpflow.probability_distributions.Gaussian(
        tf.constant([[10]], dtype=default_float()), tf.constant([[[0.1]]], dtype=default_float()),
    )
    z = gpflow.inducing_variables.InducingPoints([[-10.0], [10.0]])

    with tf.GradientTape() as tape:
        ekz = expectation(p, (kernel, z), (kernel, z))
    grad = tape.gradient(ekz, kernel.lengthscales.unconstrained_variable)
    assert grad is not None and not np.isnan(grad)


@pytest.mark.parametrize("distribution", distrs("gauss_diag"))
@pytest.mark.parametrize("kern1", kerns("rbf_act_dim_0", "lin_act_dim_0"))
@pytest.mark.parametrize("kern2", kerns("rbf_act_dim_1", "lin_act_dim_1"))
def test_eKzxKxz_separate_dims_simplification(distribution, kern1, kern2, inducing_variable):
    _check((distribution, (kern1, inducing_variable), (kern2, inducing_variable)))


@pytest.mark.parametrize("distribution", distr_args1)
@pytest.mark.parametrize("kern1", kerns("rbf_lin_sum"))
@pytest.mark.parametrize("kern2", kerns("rbf_lin_sum2"))
def test_eKzxKxz_different_sum_kernels(distribution, kern1, kern2, inducing_variable):
    _check((distribution, (kern1, inducing_variable), (kern2, inducing_variable)))


@pytest.mark.parametrize("distribution", distr_args1)
@pytest.mark.parametrize("kern1", kerns("rbf_lin_sum2"))
@pytest.mark.parametrize("kern2", kerns("rbf_lin_sum2"))
def test_eKzxKxz_same_vs_different_sum_kernels(distribution, kern1, kern2, inducing_variable):
    # check the result is the same if we pass different objects with the same value
    same = expectation(*(distribution, (kern1, inducing_variable), (kern1, inducing_variable)))
    different = expectation(*(distribution, (kern1, inducing_variable), (kern2, inducing_variable)))
    assert_allclose(same, different, rtol=RTOL)


@pytest.mark.parametrize("distribution", distrs("markov_gauss"))
@pytest.mark.parametrize("kernel", kern_args2)
@pytest.mark.parametrize("mean", means("identity"))
def test_exKxz_markov(distribution, kernel, mean, inducing_variable):
    _check((distribution, (kernel, inducing_variable), mean))


@pytest.mark.parametrize("distribution", distrs("dirac_markov_gauss"))
@pytest.mark.parametrize("kernel", kern_args2)
@pytest.mark.parametrize("mean", means("identity"))
def test_exKxz_markov_no_uncertainty(distribution, kernel, mean, inducing_variable):
    exKxz = expectation(distribution, (kernel, inducing_variable), mean)
    Kzx = kernel(Xmu_markov[:-1, :], Z)  # NxM
    xKxz = Kzx[..., None] * Xmu_markov[1:, None, :]  # NxMxD
    assert_allclose(exKxz, xKxz, rtol=RTOL)


@pytest.mark.parametrize("kernel", kerns("rbf"))
@pytest.mark.parametrize("distribution", distrs("gauss", "gauss_diag", "markov_gauss"))
def test_cov_shape_inference(distribution, kernel, inducing_variable):
    gauss_tuple = (distribution.mu, distribution.cov)
    _check((gauss_tuple, (kernel, inducing_variable)))
    if isinstance(distribution, MarkovGaussian):
        _check((gauss_tuple, None, (kernel, inducing_variable)))
