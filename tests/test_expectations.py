import numpy as np
import tensorflow as tf

import pytest
from itertools import product

import gpflow
from gpflow import settings, test_util
from gpflow.quadrature import mvnquad
from gpflow.expectations import expectation, Gaussian, get_eval_func
from gpflow.kernels import RBF
from gpflow.features import InducingPoints
from gpflow.mean_functions import Linear, Identity

def quad_expectation(p, obj1, feature1, obj2, feature2, H=150):
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature1)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    with gpflow.decors.params_as_tensors_for(p):
        return mvnquad(eval_func, p.mu, p.cov, H)

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


combinations = [
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, None, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, None, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, k2, f2),
    lambda p, k1, k2, f1, f2, m1, m2: (p, k1, f1, m1, None),
    lambda p, k1, k2, f1, f2, m1, m2: (p, m1, None, m2, None)
    ]


@pytest.mark.parametrize("p", Data.distributions)
@pytest.mark.parametrize("kern1", Data.kernels)
@pytest.mark.parametrize("feat1", Data.features)
@pytest.mark.parametrize("kern2", Data.kernels)
@pytest.mark.parametrize("feat2", Data.features)
@pytest.mark.parametrize("mean1", Data.mean_functions)
@pytest.mark.parametrize("mean2", Data.mean_functions)
@pytest.mark.parametrize("comb", combinations)
def test_kern(p, kern1, kern2, feat1, feat2, mean1, mean2, comb):
    with test_util.session_context() as sess:
        params = comb(p, kern1, kern2, feat1, feat2, mean1, mean2)
        _ = [obj.compile() for obj in params if obj is not None]

        analytic= expectation(*params)
        quad = quad_expectation(*params)
        analytic, quad = sess.run([analytic, quad])
        np.testing.assert_almost_equal(quad, analytic)

        _ = [obj.clear() for obj in params if obj is not None]
