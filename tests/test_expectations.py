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
import tensorflow as tf
import copy

import pytest

import gpflow
from gpflow.expectations import expectation, quadrature_expectation
from gpflow.probability_distributions import Gaussian, DiagonalGaussian, MarkovGaussian
from gpflow import kernels, mean_functions, features
from gpflow.test_util import session_tf
from gpflow.test_util import cache_tensor

from numpy.testing import assert_allclose

rng = np.random.RandomState(1)
RTOL = 1e-6


class Data:
    num_data = 5
    num_ind = 4
    D_in = 2
    D_out = 2

    Xmu = rng.randn(num_data, D_in)
    Xmu_markov = rng.randn(num_data + 1, D_in)  # (N+1)xD
    Xcov = rng.randn(num_data, D_in, D_in)
    Xcov = Xcov @ np.transpose(Xcov, (0, 2, 1))
    Z = rng.randn(num_ind, D_in)
    Z2 = rng.randn(num_ind - 1, D_in)
    
    cov_params = rng.randn(num_data + 1, D_in, 2 * D_in) / 2.  # (N+1)xDx2D
    NN_cov = cov_params @ np.transpose(cov_params, (0, 2, 1))  # (N+1)xDxD
    NNplus1_cross = cov_params[:-1] @ np.transpose(cov_params[1:], (0, 2, 1))  # NxDxD
    NNplus1_cross = np.concatenate((NNplus1_cross, np.zeros((1, D_in, D_in))), 0)  # (N+1)xDxD
    Xcov_markov = np.stack([NN_cov, NNplus1_cross])  # 2x(N+1)xDxD


@pytest.fixture
def feature():
    return features.InducingPoints(Data.Z)


@cache_tensor
def feature2():
    return features.InducingPoints(Data.Z2)

@cache_tensor
def gauss():
    return Gaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(Data.Xcov))


@cache_tensor
def dirac_gauss():
    return Gaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(np.zeros((Data.num_data, Data.D_in, Data.D_in))))


@cache_tensor
def gauss_diag():
    return DiagonalGaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(rng.rand(Data.num_data, Data.D_in)))


@cache_tensor
def dirac_diag():
    return DiagonalGaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(np.zeros((Data.num_data, Data.D_in))))


@cache_tensor
def markov_gauss():
    return MarkovGaussian(
        tf.convert_to_tensor(Data.Xmu_markov),
        tf.convert_to_tensor(Data.Xcov_markov))


@cache_tensor
def dirac_markov_gauss():
    return MarkovGaussian(
        tf.convert_to_tensor(Data.Xmu_markov),
        tf.convert_to_tensor(np.zeros((2, Data.num_data + 1, Data.D_in, Data.D_in))))


@cache_tensor
def rbf_kern():
    return kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand() + 1.)

@cache_tensor
def rbf_kern_2():
    # Additional cached rbf kernel for rbf cross covariance tests 
    return kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand() + 1.)

@cache_tensor
def lin_kern():
    return kernels.Linear(Data.D_in, variance=rng.rand())


@cache_tensor
def matern_kern():
    return kernels.Matern32(Data.D_in, variance=rng.rand())


@cache_tensor
def rbf_lin_sum_kern():
    return kernels.Sum([
        kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand() + 1.),
        kernels.Linear(Data.D_in, variance=rng.rand())
    ])


@cache_tensor
def rbf_lin_sum_kern2():
    return kernels.Sum([
        kernels.Linear(Data.D_in, variance=rng.rand()),
        kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand() + 1.),
        kernels.Linear(Data.D_in, variance=rng.rand()),
        kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand() + 1.),
    ])


@cache_tensor
def rbf_lin_prod_kern():
    return kernels.Product([
        kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand() + 1., active_dims=[0]),
        kernels.Linear(1, variance=rng.rand(), active_dims=[1])
    ])


@cache_tensor
def rbf_kern_act_dim_0():
    return kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand() + 1., active_dims=[0])


@cache_tensor
def rbf_kern_act_dim_1():
    return kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand() + 1., active_dims=[1])


@cache_tensor
def lin_kern_act_dim_0():
    return kernels.Linear(1, variance=rng.rand(), active_dims=[0])


@cache_tensor
def lin_kern_act_dim_1():
    return kernels.Linear(1, variance=rng.rand(), active_dims=[1])


@cache_tensor
def lin_mean():
    return mean_functions.Linear(A=rng.randn(Data.D_in, Data.D_out), b=rng.randn(Data.D_out))


@cache_tensor
def identity_mean():
    # Note: Identity can only be used if Din == Dout
    return mean_functions.Identity(input_dim=Data.D_in)


@cache_tensor
def const_mean():
    return mean_functions.Constant(c=rng.randn(Data.D_out))


@cache_tensor
def zero_mean():
    return mean_functions.Zero(output_dim=Data.D_out)


def _check(params):
    analytic = expectation(*params)
    quad = quadrature_expectation(*params)
    session = tf.get_default_session()
    analytic, quad = session.run([analytic, quad])
    assert_allclose(analytic, quad, rtol=RTOL)


# =================================== TESTS ===================================

@pytest.mark.parametrize("distribution", [gauss])
@pytest.mark.parametrize("mean1", [lin_mean, identity_mean, const_mean, zero_mean])
@pytest.mark.parametrize("mean2", [lin_mean, identity_mean, const_mean, zero_mean])
@pytest.mark.parametrize("arg_filter",
                         [lambda p, m1, m2: (p, m1),
                          lambda p, m1, m2: (p, m1, m2)])
def test_mean_function_only_expectations(session_tf, distribution, mean1, mean2, arg_filter):
    params = arg_filter(distribution(), mean1(), mean2())
    _check(params)


@pytest.mark.parametrize("distribution", [gauss, gauss_diag])
@pytest.mark.parametrize("kernel", [lin_kern, rbf_kern, rbf_lin_sum_kern, rbf_lin_prod_kern])
@pytest.mark.parametrize("arg_filter",
                         [lambda p, k, f: (p, k),
                          lambda p, k, f: (p, (k, f)),
                          lambda p, k, f: (p, (k, f), (k, f))])
def test_kernel_only_expectations(session_tf, distribution, kernel, feature, arg_filter):
    params = arg_filter(distribution(), kernel(), feature)
    _check(params)


@pytest.mark.parametrize("distribution", [gauss])
@pytest.mark.parametrize("kernel", [rbf_kern, lin_kern, matern_kern, rbf_lin_sum_kern])
@pytest.mark.parametrize("mean", [lin_mean, identity_mean, const_mean, zero_mean])
@pytest.mark.parametrize("arg_filter",
                         [lambda p, k, f, m: (p, (k, f), m),
                          lambda p, k, f, m: (p, m, (k, f))])
def test_kernel_mean_function_expectations(
        session_tf, distribution, kernel, feature, mean, arg_filter):
    params = arg_filter(distribution(), kernel(), feature, mean())
    _check(params)


@pytest.mark.parametrize("kernel", [lin_kern, rbf_kern, rbf_lin_sum_kern, rbf_lin_prod_kern])
def test_eKdiag_no_uncertainty(session_tf, kernel):
    eKdiag = expectation(dirac_diag(), kernel())
    Kdiag = kernel().Kdiag(Data.Xmu)
    eKdiag, Kdiag = session_tf.run([eKdiag, Kdiag])
    assert_allclose(eKdiag, Kdiag, rtol=RTOL)


@pytest.mark.parametrize("kernel", [lin_kern, rbf_kern, rbf_lin_sum_kern, rbf_lin_prod_kern])
def test_eKxz_no_uncertainty(session_tf, kernel, feature):
    eKxz = expectation(dirac_diag(), (kernel(), feature))
    Kxz = kernel().K(Data.Xmu, Data.Z)
    eKxz, Kxz = session_tf.run([eKxz, Kxz])
    assert_allclose(eKxz, Kxz, rtol=RTOL)


@pytest.mark.parametrize("kernel", [lin_kern, rbf_kern, rbf_lin_sum_kern])
@pytest.mark.parametrize("mean", [lin_mean, identity_mean, const_mean, zero_mean])
def test_eMxKxz_no_uncertainty(session_tf, kernel, feature, mean):
    exKxz = expectation(dirac_diag(), mean(), (kernel(), feature))
    Kxz = kernel().K(Data.Xmu, Data.Z)
    xKxz = expectation(dirac_gauss(), mean())[:, :, None] * Kxz[:, None, :]
    exKxz, xKxz = session_tf.run([exKxz, xKxz])
    assert_allclose(exKxz, xKxz, rtol=RTOL)


@pytest.mark.parametrize("kernel", [lin_kern, rbf_kern, rbf_lin_sum_kern, rbf_lin_prod_kern])
def test_eKzxKxz_no_uncertainty(session_tf, kernel, feature):
    kern = kernel()
    eKzxKxz = expectation(dirac_diag(), (kern, feature), (kern, feature))
    Kxz = kern.K(Data.Xmu, Data.Z)
    eKzxKxz, Kxz = session_tf.run([eKzxKxz, Kxz])
    KzxKxz = Kxz[:, :, None] * Kxz[:, None, :]
    assert_allclose(eKzxKxz, KzxKxz, rtol=RTOL)


def test_RBF_eKzxKxz_gradient_not_NaN(session_tf):
    """
    Ensure that <K_{Z, x} K_{x, Z}>_p(x) is not NaN and correct, when
    K_{Z, Z} is zero with finite precision. See pull request #595.
    """
    kern = gpflow.kernels.RBF(1, lengthscales=0.1)
    kern.variance = 2.

    p = gpflow.probability_distributions.Gaussian(
        tf.constant([[10]], dtype=gpflow.settings.tf_float),
        tf.constant([[[0.1]]], dtype=gpflow.settings.tf_float))
    z = gpflow.features.InducingPoints([[-10.], [10.]])

    ekz = expectation(p, (kern, z), (kern, z))

    g, = tf.gradients(ekz, kern.lengthscales._unconstrained_tensor)
    grad = session_tf.run(g)
    assert grad is not None and not np.isnan(grad)


@pytest.mark.parametrize("kernel1", [rbf_kern_act_dim_0, lin_kern_act_dim_0])
@pytest.mark.parametrize("kernel2", [rbf_kern_act_dim_1, lin_kern_act_dim_1])
def test_eKzxKxz_separate_dims_simplification(
        session_tf, kernel1, kernel2, feature):
    _check((gauss_diag(), (kernel1(), feature), (kernel2(), feature)))


def test_eKzxKxz_different_sum_kernels(session_tf, feature):
    kern1, kern2 = rbf_lin_sum_kern(), rbf_lin_sum_kern2()
    _check((gauss(), (kern1, feature), (kern2, feature)))


def test_eKzxKxz_same_vs_different_sum_kernels(session_tf, feature):
    # check the result is the same if we pass different objects with the same value
    kern1 = rbf_lin_sum_kern2()
    kern2 = copy.copy(rbf_lin_sum_kern2())
    same = expectation(*(gauss(), (kern1, feature), (kern1, feature)))
    different = expectation(*(gauss(), (kern1, feature), (kern2, feature)))
    session = tf.get_default_session()
    same, different = session.run([same, different])
    assert_allclose(same, different, rtol=RTOL)


@pytest.mark.parametrize("kernel", [rbf_kern, lin_kern, rbf_lin_sum_kern])
def test_exKxz_markov(session_tf, kernel, feature):
    _check((markov_gauss(), (kernel(), feature), identity_mean()))


@pytest.mark.parametrize("kernel", [rbf_kern, lin_kern, rbf_lin_sum_kern])
def test_exKxz_markov_no_uncertainty(session_tf, kernel, feature):
    exKxz = expectation(dirac_markov_gauss(), (kernel(), feature), identity_mean())
    exKxz = session_tf.run(exKxz)
    Kzx = kernel().compute_K(Data.Xmu_markov[:-1, :], Data.Z)  # NxM
    xKxz = Kzx[..., None] * Data.Xmu_markov[1:, None, :]  # NxMxD
    assert_allclose(exKxz, xKxz, rtol=RTOL)


@pytest.mark.parametrize("distribution", [gauss, gauss_diag, markov_gauss])
def test_cov_shape_inference(session_tf, distribution, feature):
    gauss_tuple = (distribution().mu, distribution().cov)
    _check((gauss_tuple, (rbf_kern(), feature)))
    if isinstance(distribution(), MarkovGaussian):
        _check((gauss_tuple, None, (rbf_kern(), feature)))


@pytest.mark.parametrize("distribution", [gauss, gauss_diag])
@pytest.mark.parametrize("kernel1", [rbf_kern, rbf_kern_2])
@pytest.mark.parametrize("kernel2", [rbf_kern, rbf_kern_2])
@pytest.mark.parametrize("feat1", [feature, feature2])
@pytest.mark.parametrize("feat2", [feature, feature2])
def test_eKzxKxz_rbf_cross_covariance(session_tf,
                                      distribution, kernel1, kernel2,
                                      feat1, feat2):
    _check((distribution(), (kernel1(), feat1()), (kernel2(), feat2())))
