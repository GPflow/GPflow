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

from gpflow.multioutput.conditionals import _mix_latent_gp


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
        m, v = session_tf.run([mean_tf, cov_tf], {x: X})
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


@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_broadcasting_mixing(session_tf, full_cov, full_output_cov):
    S, N = 7, 20  # batch size, num data points
    P, L = 10, 5  # observation dimensionality, num latent GPs
    W = np.random.randn(P, L)  # mixing matrix
    g_mu = np.random.randn(S, N, L)  # mean of the L latent GPs

    g_sqrt_diag = np.array([np.tril(np.random.randn(N, N)) for _ in range(S*L)])  # [S*L, N, N]
    g_sqrt_diag = np.reshape(g_sqrt_diag, [S, L, N, N])
    g_var_diag = g_sqrt_diag @ np.transpose(g_sqrt_diag, [0, 1, 3, 2])  # [S, L, N, N]
    g_var = np.zeros([S, N, L, N, L])
    for l in range(L):
        g_var[:, :, l, :, l] = g_var_diag[:, l, :, :]  # replace diagonal elements by g_var_diag

    # reference numpy implementation for mean
    f_mu_ref = g_mu @ W.T  # [S, N, P]

    # reference numpy implementation for variance
    g_var_tmp = np.transpose(g_var, [0, 1, 3, 2, 4])  # [S, N, N, L, L]
    f_var_ref = W @ g_var_tmp @ W.T  # [S, N, N, P, P]
    f_var_ref = np.transpose(f_var_ref, [0, 1, 3, 2, 4])  # [S, N, P, N, P]

    if not full_cov:
        g_var_diag = np.array([g_var_diag[:, :, n, n] for n in range(N)])  # [N, S, L]
        g_var_diag = np.transpose(g_var_diag, [1, 0, 2])  # [S, N, L]

    # run gpflow's implementation
    f_mu, f_var = session_tf.run(_mix_latent_gp(
        tf.convert_to_tensor(W),
        tf.convert_to_tensor(g_mu),
        tf.convert_to_tensor(g_var_diag),
        full_cov,
        full_output_cov
    ))

    # we strip down f_var_ref to the elements we need
    if not full_output_cov and not full_cov:
        f_var_ref = np.array([f_var_ref[:, :, p, :, p] for p in range(P)])  # [P, S, N, N]
        f_var_ref = np.array([f_var_ref[:, :, n, n] for n in range(N)])  # [N, P, S]
        f_var_ref = np.transpose(f_var_ref, [2, 0, 1])  # [S, N, P]

    elif not full_output_cov and full_cov:
        f_var_ref = np.array([f_var_ref[:, :, p, :, p] for p in range(P)])  # [P, S, N, N]
        f_var_ref = np.transpose(f_var_ref, [1, 0, 2, 3])  # [S, P, N, N]

    elif full_output_cov and not full_cov:
        f_var_ref = np.array([f_var_ref[:, n, :, n, :] for n in range(N)])  # [N, S, P, P]
        f_var_ref = np.transpose(f_var_ref, [1, 0, 2, 3])  # [S, N, P, P]

    else:
        pass  # f_var_ref has shape [..., N, P, N, P] as expected

    # check equality for mean and variance of f
    assert_allclose(f_mu_ref, f_mu)
    assert_allclose(f_var_ref, f_var)


if __name__ == '__main__':
    tf.test.main()
