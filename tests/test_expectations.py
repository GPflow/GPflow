import numpy as np
import tensorflow as tf

import pytest

import gpflow
from gpflow import test_util
from gpflow.expectations import expectation, Gaussian, EXPECTATION_QUAD_IMPL
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

    with gpflow.decors.defer_build():
        # features
        ip = features.InducingPoints(Z)
        # kernels
        rbf = kernels.RBF(D_in, variance=rng.rand(), lengthscales=rng.rand())
        rbf2 = kernels.RBF(1, variance=rng.rand(), lengthscales=rng.rand(), active_dims=[1])
        lin_kern = kernels.Linear(D_in, variance=rng.rand())
        # add1 = rbf + rbf2
        add2 = rbf + lin_kern
        # mean functions
        lin = mean_functions.Linear(rng.rand(D_in, D_out), rng.rand(D_out))
        iden = mean_functions.Identity(D_in) # Note: Identity can only be used if Din == Dout
        zero = mean_functions.Zero(output_dim=D_out)
        const = mean_functions.Constant(rng.rand(D_out))


    distributions = [gauss]
    features = [ip]
    kernels = [add2]
    mean_functions = [lin]

filters = [
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, None, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, k2, f2),
    # lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, m1, None),
    # lambda p, k1, k2, f1, f2, m1, m2: (p, m1, None, None, None),
    # lambda p, k1, k2, f1, f2, m1, m2: (p, m1, None, m2, None)
    ]


@pytest.mark.parametrize("p", Data.distributions)
@pytest.mark.parametrize("kern", Data.kernels)
@pytest.mark.parametrize("feat", Data.features)
@pytest.mark.parametrize("mean1", Data.mean_functions)
@pytest.mark.parametrize("mean2", Data.mean_functions)
@pytest.mark.parametrize("arg_filter", filters)
def test_kern(p, kern, feat, mean1, mean2, arg_filter):
    params = arg_filter(p, kern, kern, feat, feat, mean1, mean2)

    implementation = expectation.dispatch(*map(type, params))
    if implementation == EXPECTATION_QUAD_IMPL:
        # Don't evaluate if both implementations are doing quadrature.
        # This means that there is no analytic implementation available
        # for the particular combination of parameters.
        assert False
        return

    with test_util.session_context(Data.graph) as sess:
        _ = [obj.compile() for obj in params[1:] if obj is not None]

        analytic = implementation(*params)
        quad = EXPECTATION_QUAD_IMPL(*params, H=30)
        analytic, quad = sess.run([analytic, quad])
        np.testing.assert_almost_equal(quad, analytic, decimal=2)
        _ = [obj.clear() for obj in params[1:] if obj is not None]
