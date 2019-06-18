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
This test suite will check if the conditionals broadcast correctly
when the input tensors have leading dimensions.
"""


import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
from gpflow.conditionals import conditional, sample_conditional, _rollaxis_left, _rollaxis_right
from gpflow.multioutput.conditionals import _mix_latent_gp
from gpflow.test_util import session_tf


class Data:
    S1, S2, N, M = 7, 6, 4, 3
    Dx, Dy, L = 2, 5, 4  # input dim, output dim, observation dimensionality i.e. num latent GPs
    W = np.random.randn(Dy, L)  # mixing matrix

    SX = np.random.randn(S1*S2, N, Dx)
    S1_S2_X = np.reshape(SX, [S1, S2, N, Dx])

    Z = np.random.randn(M, Dx)


@pytest.mark.parametrize("full_cov", [False, True])
@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("conditional_type", ["mixing", "Z", "inducing_points"])
def test_conditional_broadcasting(session_tf, full_cov, white, conditional_type):
    """
    Test that the `conditional` and `sample_conditional` broadcasts correctly
    over leading dimensions of Xnew. Xnew can be shape [..., N, D],
    and conditional should broadcast over the [...].
    """
    X_ = tf.placeholder(tf.float64, [None, None])
    q_mu = np.random.randn(Data.M, Data.Dy)
    q_sqrt = np.tril(np.random.randn(Data.Dy, Data.M, Data.M), -1)

    if conditional_type == "Z":
        feat = Data.Z
        kern = gpflow.kernels.Matern52(Data.Dx, lengthscales=0.5)
    elif conditional_type == "inducing_points":
        feat = gpflow.features.InducingPoints(Data.Z)
        kern = gpflow.kernels.Matern52(Data.Dx, lengthscales=0.5)
    elif conditional_type == "mixing":
        # variational params have different output dim in this case
        q_mu = np.random.randn(Data.M, Data.L)
        q_sqrt = np.tril(np.random.randn(Data.L, Data.M, Data.M), -1)
        feat = mf.MixedKernelSharedMof(gpflow.features.InducingPoints(Data.Z))
        kern = mk.SeparateMixedMok(
            kernels=[gpflow.kernels.Matern52(Data.Dx, lengthscales=0.5) for _ in range(Data.L)],
            W=Data.W
        )

    if conditional_type == "mixing" and full_cov:
        pytest.skip("combination is not implemented")

    num_samples = 5
    sample_tf, mean_tf, cov_tf = sample_conditional(
        X_,
        feat,
        kern,
        tf.convert_to_tensor(q_mu),
        q_sqrt=tf.convert_to_tensor(q_sqrt),
        white=white,
        full_cov=full_cov,
        num_samples=num_samples
    )

    ss, ms, vs = [], [], []
    for X in Data.SX:
        s, m, v = session_tf.run([sample_tf, mean_tf, cov_tf], {X_: X})
        ms.append(m)
        vs.append(v)
        ss.append(s)

    ms = np.array(ms)
    vs = np.array(vs)
    ss = np.array(ss)

    ss_S12, ms_S12, vs_S12 = session_tf.run(
        sample_conditional(
            Data.SX,
            feat,
            kern,
            tf.convert_to_tensor(q_mu),
            q_sqrt=tf.convert_to_tensor(q_sqrt),
            white=white,
            full_cov=full_cov,
            num_samples=num_samples
        )
    )

    ss_S1_S2, ms_S1_S2, vs_S1_S2 = session_tf.run(
        sample_conditional(
            Data.S1_S2_X,
            feat,
            kern,
            tf.convert_to_tensor(q_mu),
            q_sqrt=tf.convert_to_tensor(q_sqrt),
            white=white,
            full_cov=full_cov,
            num_samples=num_samples
        )
    )

    assert_allclose(ss_S12.shape, ss.shape)
    assert_allclose(ms_S12, ms)
    assert_allclose(vs_S12, vs)
    assert_allclose(ms_S1_S2.reshape(Data.S1 * Data.S2, Data.N, Data.Dy), ms)
    assert_allclose(ss_S1_S2.shape, [Data.S1, Data.S2, num_samples, Data.N, Data.Dy])

    if full_cov:
        assert_allclose(vs_S1_S2.reshape(Data.S1 * Data.S2, Data.Dy, Data.N, Data.N), vs)
    else:
        assert_allclose(vs_S1_S2.reshape(Data.S1 * Data.S2, Data.N, Data.Dy), vs)


# -------------------------------------------
# Test utility functions used in conditionals
# -------------------------------------------

# _mix_latent_gps
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_broadcasting_mix_latent_gps(session_tf, full_cov, full_output_cov):
    S, N = 7, 20  # batch size, num data points
    P, L = 10, 5  # observation dimensionality, num latent GPs
    W = np.random.randn(P, L)  # mixing matrix
    g_mu = np.random.randn(S, N, L)  # mean of the L latent GPs

    g_sqrt_diag = np.tril(np.random.randn(S*L, N, N), -1)  # [L*S, N, N]
    g_sqrt_diag = np.reshape(g_sqrt_diag, [L, S, N, N])
    g_var_diag = g_sqrt_diag @ np.transpose(g_sqrt_diag, [0, 1, 3, 2])  # [L, S, N, N]
    g_var = np.zeros([S, N, L, N, L])
    for l in range(L):
        g_var[:, :, l, :, l] = g_var_diag[l, :, :, :]  # replace diagonal elements by g_var_diag

    # reference numpy implementation for mean
    f_mu_ref = g_mu @ W.T  # [S, N, P]

    # reference numpy implementation for variance
    g_var_tmp = np.transpose(g_var, [0, 1, 3, 2, 4])  # [S, N, N, L, L]
    f_var_ref = W @ g_var_tmp @ W.T  # [S, N, N, P, P]
    f_var_ref = np.transpose(f_var_ref, [0, 1, 3, 2, 4])  # [S, N, P, N, P]

    if not full_cov:
        g_var_diag = np.array([g_var_diag[:, :, n, n] for n in range(N)])  # [N, L, S]
        g_var_diag = np.transpose(g_var_diag, [2, 0, 1])  # [S, N, L]

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


# rollaxis
@pytest.mark.parametrize("rolls", [1, 2])
@pytest.mark.parametrize("direction", ["left", "right"])
def test_rollaxis(session_tf, rolls, direction):
    A = np.random.randn(10, 5, 3)
    A_tf = tf.convert_to_tensor(A)

    if direction == "left":
        perm = [1, 2, 0] if rolls == 1 else [2, 0, 1]
    elif direction == "right":
        perm = [2, 0, 1] if rolls == 1 else [1, 2, 0]

    A_rolled_ref = np.transpose(A, perm)

    if direction == "left":
        A_rolled_tf = _rollaxis_left(A_tf, rolls)
    elif direction == "right":
        A_rolled_tf = _rollaxis_right(A_tf, rolls)

    A_rolled_tf = session_tf.run(A_rolled_tf)
    assert_allclose(A_rolled_ref, A_rolled_tf)


@pytest.mark.parametrize("rolls", [1, 2])
def test_rollaxis_idempotent(session_tf, rolls):
    A = np.random.randn(10, 5, 3, 20, 1)
    A_tf = tf.convert_to_tensor(A)
    A_left_right = session_tf.run(_rollaxis_left(_rollaxis_right(A_tf, 2), 2))
    A_right_left = session_tf.run(_rollaxis_right(_rollaxis_left(A_tf, 2), 2))

    assert_allclose(A, A_left_right)
    assert_allclose(A, A_right_left)
