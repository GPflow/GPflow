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
# limitations under the License.


"""
This test suite will check if the conditionals broadcast correct
when the input tensors have a leading dimensions.
"""


import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.test_util import session_tf
from gpflow.conditionals import conditional


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("features_inducing_points", [False, True])
def test_base_conditional_leading_dims(session_tf, full_cov, white, features_inducing_points):
    """
    Test that the conditional broadcasts correctly over leading dimensions of Xnew
    Xnew can be shape [..., N, D], and conditional should broadcast over the [...]
    """
    S1, S2, Dy, N, M, Dx = 7, 6, 5, 4, 3, 2

    SX = np.random.randn(S1*S2, N, Dx)
    S1_S2_X = np.reshape(SX, [S1, S2, N, Dx])
    Z = np.random.randn(M, Dx)
    if features_inducing_points:
        Z = gpflow.features.InducingPoints(Z)

    kern = gpflow.kernels.Matern52(Dx, lengthscales=0.5)

    q_mu = np.random.randn(M, Dy)
    q_sqrt = np.tril(np.random.randn(Dy, M, M), -1)

    x = tf.placeholder(tf.float64, [None, None])

    mean_tf, cov_tf = conditional(
        x, 
        Z, 
        kern, 
        q_mu, 
        q_sqrt=tf.identity(q_sqrt),
        white=white,
        full_cov=full_cov
    )

    ms, vs = [], []
    for X in SX:
        m, v = session_tf.run([mean_tf, cov_tf], {x:X})
        ms.append(m)
        vs.append(v)

    ms = np.array(ms)
    vs = np.array(vs)

    ms_S12, vs_S12 = session_tf.run(conditional(
        SX,
        Z,
        kern,
        q_mu,
        q_sqrt=tf.convert_to_tensor(q_sqrt),
        white=white,
        full_cov=full_cov
    ))

    ms_S1_S2, vs_S1_S2 = session_tf.run(conditional(
        S1_S2_X,
        Z,
        kern,
        q_mu,
        q_sqrt=tf.convert_to_tensor(q_sqrt),
        white=white,
        full_cov=full_cov
    ))

    assert_allclose(ms_S12, ms)
    assert_allclose(vs_S12, vs)
    assert_allclose(ms_S1_S2.reshape(S1 * S2, N, Dy), ms)

    if full_cov:
        assert_allclose(vs_S1_S2.reshape(S1 * S2, Dy, N, N), vs)
    else:
        assert_allclose(vs_S1_S2.reshape(S1 * S2, N, Dy), vs)


if __name__ == '__main__':
    tf.test.main()
