import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.test_util import session_context

def arr(x):
    return tf.cast(np.asarray(x), dtype=gpflow.settings.float_type)

def test_diagquad_2d():
    with session_context() as session:
        quad = gpflow.quadrature.ndiagquad(
                lambda *X: X[0] + X[1], 25,
                [arr([0.0]), arr([0.0])],
                [arr([1.0]), arr([1.0])])
        res = session.run(quad)
        assert_allclose(res, np.array([0.0]), atol=1e-10)


