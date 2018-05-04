import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.test_util import session_context

@pytest.fixture
def mu1(): return np.array([1.0])

@pytest.fixture
def mu2(): return np.array([2.0])

@pytest.fixture
def var1(): return np.array([3.0])

@pytest.fixture
def var2(): return np.array([4.0])

def cast(x):
    return tf.cast(np.asarray(x), dtype=gpflow.settings.float_type)

def test_diagquad_1d(mu1, var1):
    with session_context() as session:
        quad = gpflow.quadrature.ndiagquad(
                lambda *X: tf.exp(X[0]), 25,
                [cast(mu1)], [cast(var1)])
        res = session.run(quad)
        expected = np.exp(mu1 + var1/2)
        assert_allclose(res, expected, atol=1e-10)


def test_diagquad_2d(mu1, var1, mu2, var2):
    with session_context() as session:
        quad = gpflow.quadrature.ndiagquad(
                lambda *X: tf.exp(X[0]), 25,
                [cast(mu1), cast(mu2)], [cast(var1), cast(var2)])
        res = session.run(quad)
        expected = np.exp(mu1 + var1/2)
        assert_allclose(res, expected, atol=1e-10)


