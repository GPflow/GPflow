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

from functools import partial


def gen_L(rng, n, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])


class Data:
    rng = np.random.RandomState(1)
    num_data = 5
    num_ind = 4
    D_in = 2
    D_out = 2

    Xmu = rng.randn(num_data, D_in)
    L = gen_L(rng, num_data, D_in, D_in)
    Xvar = np.array([l @ l.T for l in L])
    Z = rng.randn(num_ind, D_in)

    # distributions don't need to be compiled (No Parameter objects)
    # but the members should be Tensors created in the same graph
    graph = tf.Graph()
    with test_util.session_context(graph) as sess:
        gauss = Gaussian(tf.constant(Xmu), tf.constant(Xvar))
        dirac = Gaussian(tf.constant(Xmu), tf.constant(np.zeros((num_data, D_in, D_in))))
        gauss_diag = DiagonalGaussian(tf.constant(Xmu), tf.constant(rng.rand(num_data, D_in)))
        dirac_diag = DiagonalGaussian(tf.constant(Xmu), tf.constant(np.zeros((num_data, D_in))))
        dirac_markov_gauss = MarkovGaussian(tf.constant(Xmu), tf.constant(np.zeros((2, num_data, D_in, D_in))))

        # create the covariance for the pairwise markov-gaussian
        dummy_gen = lambda rng, n, *shape: np.array([rng.randn(*shape) for _ in range(n)])
        L_mg = dummy_gen(rng, num_data, D_in, 2*D_in)
        LL = np.concatenate((L_mg[:-1], L_mg[1:]), 1)
        Xcov = LL @ np.transpose(LL, (0, 2, 1))
        Xc = np.concatenate((Xcov[:, :D_in, :D_in], Xcov[-1:, D_in:, D_in:]), 0)
        Xcross = np.concatenate((Xcov[:, :D_in, D_in:], np.zeros((1, D_in, D_in))), 0)
        Xcc = np.stack([Xc, Xcross])

        markov_gauss = MarkovGaussian(Xmu, Xcc)

    with gpflow.decors.defer_build():
        # features
        ip = features.InducingPoints(Z)
        # kernels
        rbf_prod_seperate_dims = kernels.Product([
            kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand(), active_dims=[0]),
            kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand(), active_dims=[1])
            ])

        rbf_lin_sum = kernels.Sum([
            kernels.RBF(D_in, variance=rng.rand(), lengthscales=rng.rand()),
            kernels.RBF(D_in, variance=rng.rand(), lengthscales=rng.rand()),
            kernels.Linear(D_in, variance=rng.rand())
            ])

        rbf = kernels.RBF(D_in, variance=rng.rand(), lengthscales=rng.rand())

        lin_kern = kernels.Linear(D_in, variance=rng.rand())

        # mean functions
        lin = mean_functions.Linear(rng.rand(D_in, D_out), rng.rand(D_out))
        iden = mean_functions.Identity(D_in) # Note: Identity can only be used if Din == Dout
        zero = mean_functions.Zero(output_dim=D_out)
        const = mean_functions.Constant(rng.rand(D_out))


def _execute_func_on_params(params, func_name):
    # This construction just flattens a list consisting of objects and tuple of objects.
    # The member function `func_name` is executed for each element of the list.
    _ = [getattr(param, func_name)() for param_tuple in params
            for param in (param_tuple if isinstance(param_tuple, tuple) else (param_tuple,))]


def _test(params):
    _execute_func_on_params(params[1:], 'compile')

    analytic = expectation(*params)
    quad = quadrature_expectation(*params)
    analytic, quad = tf.get_default_session().run([analytic, quad])
    np.testing.assert_almost_equal(quad, analytic, decimal=2)

    _execute_func_on_params(params[1:], 'clear')


@pytest.mark.parametrize("distribution", [Data.gauss, Data.gauss_diag])
@pytest.mark.parametrize("kern", [Data.lin_kern, Data.rbf, Data.rbf_lin_sum, Data.rbf_prod_seperate_dims])
@pytest.mark.parametrize("feat", [Data.ip])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f: (p, k),
                            lambda p, k, f: (p, (f, k)),
                            lambda p, k, f: (p, (f, k), (f, k))])
@test_util.session_context(Data.graph)
def test_psi_stats(distribution, kern, feat, arg_filter):
    params = arg_filter(distribution, kern, feat)
    _test(params)


@pytest.mark.parametrize("distribution", [Data.gauss])
@pytest.mark.parametrize("mean1", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("mean2", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, m1, m2: (p, m1),
                            lambda p, m1, m2: (p, m1, m2)])
@test_util.session_context(Data.graph)
def test_mean_function_expectations(distribution, mean1, mean2, arg_filter):
    params = arg_filter(distribution, mean1, mean2)
    _test(params)


@pytest.mark.parametrize("distribution", [Data.gauss])
@pytest.mark.parametrize("mean", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern])
@pytest.mark.parametrize("feat", [Data.ip])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f, m: (p, (f, k), m),
                            lambda p, k, f, m: (p, m, (f, k))])
@test_util.session_context(Data.graph)
def test_kernel_mean_function_expectation(distribution, mean, kern, feat, arg_filter):
    params = arg_filter(distribution, kern, feat, mean)
    _test(params)


def _compile_params(kern, feat):
    kern.compile()
    feat.compile()
    return kern, feat


def _clear_params(kern, feat):
    kern.clear()
    feat.clear()


@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern])
@test_util.session_context(graph=Data.graph)
def test_eKdiag_no_uncertainty(kern):
    kern, _ = _compile_params(kern, Data.ip)
    eKdiag = expectation(Data.dirac, kern)
    Kdiag = kern.Kdiag(Data.Xmu)
    eKdiag, Kdiag = tf.get_default_session().run([eKdiag, Kdiag])
    np.testing.assert_almost_equal(eKdiag, Kdiag)
    _clear_params(kern, _)


@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern])
@test_util.session_context(graph=Data.graph)
def test_eKxz_no_uncertainty(kern):
    kern, feat = _compile_params(kern, Data.ip)
    eKxz = expectation(Data.dirac, (feat, kern))
    Kxz = kern.K(Data.Xmu, Data.Z)
    eKxz, Kxz = tf.get_default_session().run([eKxz, Kxz])
    np.testing.assert_almost_equal(eKxz, Kxz)
    _clear_params(kern, feat)


@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern])
@test_util.session_context(graph=Data.graph)
def test_eKxzzx_no_uncertainty(kern):
    kern, feat = _compile_params(kern, Data.ip)
    eKxzzx = expectation(Data.dirac, (feat, kern), (feat, kern))
    Kxz = kern.K(Data.Xmu, Data.Z)
    eKxzzx, Kxz = tf.get_default_session().run([eKxzzx, Kxz])
    Kxzzx = Kxz[:, :, None] * Kxz[:, None, :]
    np.testing.assert_almost_equal(eKxzzx, Kxzzx)
    _clear_params(kern, feat)


@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern, Data.rbf_lin_sum])
@test_util.session_context(graph=Data.graph)
def test_exKxz_pairwise_no_uncertainty(kern):
    kern, feat = _compile_params(kern, Data.ip)
    exKxz_pairwise = expectation(Data.dirac_markov_gauss, (feat, kern), Data.iden)
    exKxz_pairwise = tf.get_default_session().run(exKxz_pairwise)
    Kxz = kern.compute_K(Data.Xmu[:-1, :], Data.Z)  # NxM
    xKxz_pairwise = np.einsum('nm,nd->nmd', Kxz, Data.Xmu[1:, :])
    np.testing.assert_almost_equal(exKxz_pairwise, xKxz_pairwise)
    _clear_params(kern, feat)


@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern, Data.rbf_lin_sum])
@test_util.session_context(graph=Data.graph)
def test_exKxz_pairwise(kern):
    _test((Data.markov_gauss, (Data.ip, kern), Data.iden))
