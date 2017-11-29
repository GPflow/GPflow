import numpy as np
import tensorflow as tf

import pytest
from itertools import product

import gpflow
from gpflow import settings, test_util
from gpflow.expectations import expectation, Gaussian, get_eval_func
from gpflow.kernels import RBF
from gpflow.features import InducingPoints
from gpflow.mean_functions import Linear, Identity

def gen_L(rng, n, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])

rng = np.random.RandomState(1)

class Data:
    num_data = 5
    num_ind = 4
    D_in = 2
    D_out = 3

    Xmu = rng.randn(num_data, D_in)
    L = gen_L(rng, num_data, D_in, D_in)
    Xvar = np.array([l @ l.T for l in L])
    Z = rng.randn(num_ind, D_in)

    with gpflow.decors.defer_build():
        gauss = Gaussian(Xmu, Xvar)
        ip = InducingPoints(Z)
        rbf = RBF(D_in, variance=rng.rand(), lengthscales=rng.rand())
        rbf2 = RBF(D_in, variance=rng.rand(), lengthscales=rng.rand())
        lin = Linear(rng.rand(D_in, 1), [rng.rand()])
        iden = Identity()

        distributions = [gauss]
        features = [ip]
        kernels = [rbf]
        mean_functions = [lin, iden]


filters = [
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, None, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, k2, f2),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, m1, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, m1, None, m2, None)
    ]


quad_implementation = expectation.dispatch(Gaussian, object, type(None), object, type(None))

@pytest.mark.parametrize("p", Data.distributions)
@pytest.mark.parametrize("kern1", Data.kernels)
@pytest.mark.parametrize("feat1", Data.features)
@pytest.mark.parametrize("kern2", Data.kernels)
@pytest.mark.parametrize("feat2", Data.features)
@pytest.mark.parametrize("mean1", Data.mean_functions)
@pytest.mark.parametrize("mean2", Data.mean_functions)
@pytest.mark.parametrize("filter", filters)
def test_kern(p, kern1, kern2, feat1, feat2, mean1, mean2, filter):
    params = filter(p, kern1, kern2, feat1, feat2, mean1, mean2)

    implementation = expectation.dispatch(*map(type, params))
    if implementation == quad_implementation:
        # Don't evaluate if both implementations are doing quadrature
        return

    with test_util.session_context() as sess:
        _ = [obj.compile() for obj in params if obj is not None]
        analytic = implementation(*params)
        quad = quad_implementation(*params, H=100)
        analytic, quad = sess.run([analytic, quad])
        np.testing.assert_almost_equal(quad, analytic)

        _ = [obj.clear() for obj in params if obj is not None]


