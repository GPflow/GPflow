# Copyright 2017 the GPflow authors.
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
# limitations under the License.from __future__ import print_function

import numpy as np
import tensorflow as tf

import pytest

import gpflow
from gpflow import test_util
from gpflow.expectations import expectation
from gpflow.expectations_quadrature import quadrature_expectation
from gpflow.probability_distributions import Gaussian, DiagonalGaussian, MarkovGaussian
from gpflow import kernels, mean_functions, features
from gpflow.test_util import session_tf
from gpflow.test_util import cache_tensor

from numpy.testing import assert_allclose


rng = np.random.RandomState(1)


@pytest.fixture
def feature(session_tf):
    return features.InducingPoints(Data.Z)


def gen_L(n, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])


class Data:
    num_data = 5
    num_ind = 4
    D_in = 2
    D_out = 2

    Xmu = rng.randn(num_data, D_in)
    L = gen_L(num_data, D_in, D_in)
    Xvar = np.array([l @ l.T for l in L])
    Z = rng.randn(num_ind, D_in)


@cache_tensor
def gauss():
    return Gaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(Data.Xvar))


@cache_tensor
def dirac():
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
def dirac_markov_gauss():
    return MarkovGaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(np.zeros((2, Data.num_data, Data.D_in, Data.D_in))))


@cache_tensor
def markov_gauss():
    # create the covariance for the pairwise markov-gaussian
    dummy_gen = lambda rng, n, *shape: np.array([rng.randn(*shape) for _ in range(n)])
    D_in = Data.D_in
    L_mg = dummy_gen(rng, Data.num_data, D_in, 2*D_in)  # N+1 x D x 2D
    LL = np.concatenate((L_mg[:-1], L_mg[1:]), 1)  # N x 2D x 2D
    Xcov = LL @ np.transpose(LL, (0, 2, 1))
    Xc = np.concatenate((Xcov[:, :D_in, :D_in], Xcov[-1:, D_in:, D_in:]), 0)  # N+1 x D x D
    Xcross = np.concatenate((Xcov[:, :D_in, D_in:], np.zeros((1, D_in, D_in))), 0)  # N+1 x D x D
    Xcc = np.stack([Xc, Xcross])  # 2 x N+1 x D x D
    return MarkovGaussian(
        tf.convert_to_tensor(Data.Xmu),
        tf.convert_to_tensor(Xcc))


@cache_tensor
def rbf_prod_seperate_dims():
    return kernels.Product([
        kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand(), active_dims=[0]),
        kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand(), active_dims=[1])
    ])


@cache_tensor
def rbf_lin_sum():
    return kernels.Sum([
        kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand()),
        kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand()),
        kernels.Linear(Data.D_in, variance=rng.rand())
    ])


@cache_tensor
def rbf():
    return kernels.RBF(Data.D_in, variance=rng.rand(), lengthscales=rng.rand())


@cache_tensor
def lin_kern():
    return kernels.Linear(Data.D_in, variance=rng.rand())


@cache_tensor
def lin():
    return mean_functions.Linear(rng.rand(Data.D_in, Data.D_out), rng.rand(Data.D_out))


@cache_tensor
def identity():
    # Note: Identity can only be used if Din == Dout
    return mean_functions.Identity(Data.D_in)


@cache_tensor
def zero():
    return mean_functions.Zero(output_dim=Data.D_out)


@cache_tensor
def const():
    return mean_functions.Constant(rng.rand(Data.D_out))


def _check(params):
    analytic = expectation(*params)
    quad = quadrature_expectation(*params)
    session = tf.get_default_session()
    analytic, quad = session.run([analytic, quad])
    assert_allclose(analytic, quad, rtol=1e-1)


@pytest.mark.parametrize("distribution", [gauss, gauss_diag])
@pytest.mark.parametrize("kernel", [lin_kern, rbf, rbf_lin_sum, rbf_prod_seperate_dims])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f: (p, k),
                            lambda p, k, f: (p, (k, f)),
                            lambda p, k, f: (p, (k, f), (k, f))])
def test_psi_stats(session_tf, feature, distribution, kernel, arg_filter):
    params = arg_filter(distribution(), kernel(), feature)
    _check(params)


@pytest.mark.parametrize("distribution", [gauss])
@pytest.mark.parametrize("mean1", [lin, identity, const, zero])
@pytest.mark.parametrize("mean2", [lin, identity, const, zero])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, m1, m2: (p, m1),
                            lambda p, m1, m2: (p, m1, m2)])
def test_mean_function_expectations(session_tf, distribution, mean1, mean2, arg_filter):
    params = arg_filter(distribution(), mean1(), mean2())
    _check(params)


@pytest.mark.parametrize("distribution", [gauss])
@pytest.mark.parametrize("mean", [lin, identity, const, zero])
@pytest.mark.parametrize("kernel", [rbf, lin_kern])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f, m: (p, (k, f), m),
                            lambda p, k, f, m: (p, m, (k, f))])
def test_kernel_mean_function_expectation(
        session_tf, feature, distribution, mean, kernel, arg_filter):
    params = arg_filter(distribution(), kernel(), feature, mean())
    _check(params)


@pytest.mark.parametrize("kernel", [rbf, lin_kern])
def test_eKdiag_no_uncertainty(session_tf, kernel):
    eKdiag = expectation(dirac(), kernel())
    Kdiag = kernel().Kdiag(Data.Xmu)
    eKdiag, Kdiag = session_tf.run([eKdiag, Kdiag])
    assert_allclose(eKdiag, Kdiag, rtol=0.)


@pytest.mark.parametrize("kernel", [rbf, lin_kern])
def test_eKxz_no_uncertainty(session_tf, kernel, feature):
    eKxz = expectation(dirac(), (feature, kernel()))
    Kxz = kernel().K(Data.Xmu, Data.Z)
    eKxz, Kxz = session_tf.run([eKxz, Kxz])
    assert_allclose(eKxz, Kxz, rtol=0.)


@pytest.mark.parametrize("kernel", [rbf, lin_kern])
def test_eKxzzx_no_uncertainty(session_tf, kernel, feature):
    eKxzzx = expectation(dirac(), (feature, kernel()), (feature, kernel()))
    Kxz = kernel().K(Data.Xmu, Data.Z)
    eKxzzx, Kxz = session_tf.run([eKxzzx, Kxz])
    Kxzzx = Kxz[:, :, None] * Kxz[:, None, :]
    assert_allclose(eKxzzx, Kxzzx, rtol=0.)


@pytest.mark.parametrize("kernel", [rbf, lin_kern, rbf_lin_sum])
def test_exKxz_pairwise_no_uncertainty(session_tf, kernel, feature):
    exKxz_pairwise = expectation(dirac_markov_gauss(), (feature, kernel()), identity())
    exKxz_pairwise = session_tf.run(exKxz_pairwise)
    Kxz = kernel().compute_K(Data.Xmu[:-1, :], Data.Z)  # NxM
    xKxz_pairwise = np.einsum('nm,nd->nmd', Kxz, Data.Xmu[1:, :])
    assert_allclose(exKxz_pairwise, xKxz_pairwise, rtol=0.)


@pytest.mark.parametrize("kernel", [rbf, lin_kern, rbf_lin_sum])
def test_exKxz_pairwise(session_tf, kernel, feature):
    _check((markov_gauss(), (kernel(), feature), identity()))
