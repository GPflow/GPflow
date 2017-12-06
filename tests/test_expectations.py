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
from gpflow.expectations_quadrature import EXPECTATION_QUAD_IMPL
from gpflow.probability_distributions import Gaussian
from gpflow import kernels, mean_functions, features


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
    # This construction just flattens a list of objects and tuple of objects.
    # The member function `func_name` is executed for each element of the list.
    _ = [getattr(param, func_name)() for param_tuple in params
            for param in (param_tuple if isinstance(param_tuple, tuple) else (param_tuple,))]


def _quadrature_expectation(params):
    if len(params) == 2:
        if isinstance(params[1], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], *params[1], None, None)
        elif not isinstance(params[1], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], params[1], None, None, None)
    elif len(params) == 3:
        if isinstance(params[1], tuple) and isinstance(params[2], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], *params[1], *params[2])
        elif isinstance(params[1], tuple) and not isinstance(params[2], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], *params[1], params[2], None)
        elif not isinstance(params[1], tuple) and isinstance(params[2], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], params[1], None, *params[2])
        elif not isinstance(params[1], tuple) and not isinstance(params[2], tuple):
            return EXPECTATION_QUAD_IMPL(params[0], params[1], None, params[2], None)

    assert False

def _test(params):
    with test_util.session_context(Data.graph) as sess:
        _execute_func_on_params(params[1:], 'compile')

        analytic = expectation(*params)
        quad = _quadrature_expectation(params)
        analytic, quad = sess.run([analytic, quad])
        np.testing.assert_almost_equal(quad, analytic, decimal=2)

        _execute_func_on_params(params[1:], 'clear')


@pytest.mark.parametrize("distribution", [Data.gauss])
@pytest.mark.parametrize("kern", [Data.lin_kern])
@pytest.mark.parametrize("feat", [Data.ip])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f: (p, k),
                            lambda p, k, f: (p, (k, f)),
                            lambda p, k, f: (p, (k, f), (k, f))])
def test_psi_stats(distribution, kern, feat, arg_filter):
    params = arg_filter(distribution, kern, feat)
    _test(params)


@pytest.mark.parametrize("distribution", [Data.gauss])
@pytest.mark.parametrize("mean1", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("mean2", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, m1, m2: (p, m1),
                            lambda p, m1, m2: (p, m1, m2)])
def test_mean_function_expectations(distribution, mean1, mean2, arg_filter):
    params = arg_filter(distribution, mean1, mean2)
    _test(params)


@pytest.mark.parametrize("distribution", [Data.gauss])
@pytest.mark.parametrize("mean", [Data.lin, Data.iden, Data.const, Data.zero])
@pytest.mark.parametrize("kern", [Data.rbf, Data.lin_kern])
@pytest.mark.parametrize("feat", [Data.ip])
@pytest.mark.parametrize("arg_filter", [
                            lambda p, k, f, m: (p, (k, f), m),
                            lambda p, k, f, m: (p, m, (k, f))])
def test_kernel_mean_function_expectation(distribution, mean, kern, feat, arg_filter):
    params = arg_filter(distribution, kern, feat, mean)
    _test(params)


@pytest.fixture(params=[Data.rbf, Data.lin_kern])
def compile_params(request):
    with test_util.session_context(Data.graph) as sess:
        kern, feat = request.param, Data.ip
        kern.compile()
        feat.compile()
        yield sess, kern, feat
        kern.clear()
        feat.clear()


def test_eKdiag_no_uncertainty(compile_params):
    sess, kern, _ = compile_params
    eKdiag = expectation(Data.dirac, kern)
    Kdiag = kern.Kdiag(Data.Xmu)
    eKdiag, Kdiag = sess.run([eKdiag, Kdiag])
    np.testing.assert_almost_equal(eKdiag, Kdiag)


def test_eKxz_no_uncertainty(compile_params):
    sess, kern, feat = compile_params
    eKxz = expectation(Data.dirac, (kern, feat))
    Kxz = kern.K(Data.Xmu, Data.Z)
    eKxz, Kxz = sess.run([eKxz, Kxz])
    np.testing.assert_almost_equal(eKxz, Kxz)


def test_eKxzzx_no_uncertainty(compile_params):
    sess, kern, feat = compile_params
    eKxzzx = expectation(Data.dirac, (kern, feat), (kern, feat))
    Kxz = kern.K(Data.Xmu, Data.Z)
    eKxzzx, Kxz = sess.run([eKxzzx, Kxz])
    Kxzzx = Kxz[:, :, None] * Kxz[:, None, :]
    np.testing.assert_almost_equal(eKxzzx, Kxzzx)
