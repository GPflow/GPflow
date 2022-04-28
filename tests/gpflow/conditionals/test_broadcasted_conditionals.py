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

from typing import cast

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
import gpflow.inducing_variables.multioutput as mf
import gpflow.kernels.multioutput as mk
from gpflow.base import AnyNDArray, SamplesMeanAndVariance
from gpflow.conditionals import sample_conditional
from gpflow.conditionals.util import mix_latent_gp

# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Data:
    S1, S2, N, M = (
        7,  # num samples 1
        6,  # num samples 2
        4,  # num datapoints
        3,  # num inducing
    )
    Dx, Dy, L = (
        2,  # input dim
        5,  # output dim
        4,  # num latent GPs
    )
    W = np.random.randn(Dy, L)  # mixing matrix

    SX = np.random.randn(S1 * S2, N, Dx)
    S1_S2_X = np.reshape(SX, [S1, S2, N, Dx])

    Z = np.random.randn(M, Dx)


@pytest.mark.parametrize("full_cov", [False, True])
@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("conditional_type", ["mixing", "Z", "inducing_points"])
def test_conditional_broadcasting(full_cov: bool, white: bool, conditional_type: str) -> None:
    """
    Test that the `conditional` and `sample_conditional` broadcasts correctly
    over leading dimensions of Xnew. Xnew can be shape [..., N, D],
    and conditional should broadcast over the [...].
    """
    q_mu = np.random.randn(Data.M, Data.Dy)
    q_sqrt: AnyNDArray = np.tril(np.random.randn(Data.Dy, Data.M, Data.M), -1)

    if conditional_type == "Z":
        inducing_variable = Data.Z
        kernel = gpflow.kernels.Matern52(lengthscales=0.5)
    elif conditional_type == "inducing_points":
        inducing_variable = gpflow.inducing_variables.InducingPoints(Data.Z)
        kernel = gpflow.kernels.Matern52(lengthscales=0.5)
    elif conditional_type == "mixing":
        # variational params have different output dim in this case
        q_mu = np.random.randn(Data.M, Data.L)
        q_sqrt = np.tril(np.random.randn(Data.L, Data.M, Data.M), -1)
        inducing_variable = mf.SharedIndependentInducingVariables(
            gpflow.inducing_variables.InducingPoints(Data.Z)
        )
        kernel = mk.LinearCoregionalization(
            kernels=[gpflow.kernels.Matern52(lengthscales=0.5) for _ in range(Data.L)],
            W=Data.W,
        )
    else:
        raise NotImplementedError

    if conditional_type == "mixing" and full_cov:
        pytest.skip("combination is not implemented")

    num_samples = 5

    def sample_conditional_fn(X: tf.Tensor) -> SamplesMeanAndVariance:
        return cast(
            SamplesMeanAndVariance,
            sample_conditional(
                X,
                inducing_variable,
                kernel,
                tf.convert_to_tensor(q_mu),
                q_sqrt=tf.convert_to_tensor(q_sqrt),
                white=white,
                full_cov=full_cov,
                num_samples=num_samples,
            ),
        )

    samples: AnyNDArray = np.array([sample_conditional_fn(X)[0] for X in Data.SX])
    means: AnyNDArray = np.array([sample_conditional_fn(X)[1] for X in Data.SX])
    variables: AnyNDArray = np.array([sample_conditional_fn(X)[2] for X in Data.SX])

    samples_S12, means_S12, vars_S12 = sample_conditional(
        Data.SX,
        inducing_variable,
        kernel,
        tf.convert_to_tensor(q_mu),
        q_sqrt=tf.convert_to_tensor(q_sqrt),
        white=white,
        full_cov=full_cov,
        num_samples=num_samples,
    )

    samples_S1_S2, means_S1_S2, vars_S1_S2 = sample_conditional(
        Data.S1_S2_X,
        inducing_variable,
        kernel,
        tf.convert_to_tensor(q_mu),
        q_sqrt=tf.convert_to_tensor(q_sqrt),
        white=white,
        full_cov=full_cov,
        num_samples=num_samples,
    )

    assert_allclose(samples_S12.shape, samples.shape)
    assert_allclose(samples_S1_S2.shape, [Data.S1, Data.S2, num_samples, Data.N, Data.Dy])
    assert_allclose(means_S12, means)
    assert_allclose(vars_S12, variables)
    assert_allclose(means_S1_S2.numpy().reshape(Data.S1 * Data.S2, Data.N, Data.Dy), means)
    if full_cov:
        vars_s1_s2 = vars_S1_S2.numpy().reshape(Data.S1 * Data.S2, Data.Dy, Data.N, Data.N)
        assert_allclose(vars_s1_s2, variables)
    else:
        vars_s1_s2 = vars_S1_S2.numpy().reshape(Data.S1 * Data.S2, Data.N, Data.Dy)
        assert_allclose(vars_s1_s2, variables)


# -------------------------------------------
# Test utility functions used in conditionals
# -------------------------------------------

# _mix_latent_gps
@pytest.mark.parametrize("full_cov", [True, False])
@pytest.mark.parametrize("full_output_cov", [True, False])
def test_broadcasting_mix_latent_gps(full_cov: bool, full_output_cov: bool) -> None:
    S, N = 7, 20  # batch size, num data points
    P, L = 10, 5  # observation dimensionality, num latent GPs
    W = np.random.randn(P, L)  # mixing matrix
    g_mu = np.random.randn(S, N, L)  # mean of the L latent GPs

    g_sqrt_diag: AnyNDArray = np.tril(np.random.randn(S * L, N, N), -1)  # [L*S, N, N]
    g_sqrt_diag = np.reshape(g_sqrt_diag, [L, S, N, N])
    g_var_diag: AnyNDArray = g_sqrt_diag @ np.transpose(g_sqrt_diag, [0, 1, 3, 2])  # [L, S, N, N]
    g_var = np.zeros([S, N, L, N, L])
    for l in range(L):
        g_var[:, :, l, :, l] = g_var_diag[l, :, :, :]  # replace diagonal elements by g_var_diag

    # reference numpy implementation for mean
    f_mu_ref: AnyNDArray = g_mu @ W.T  # [S, N, P]

    # reference numpy implementation for variance
    g_var_tmp = np.transpose(g_var, [0, 1, 3, 2, 4])  # [S, N, N, L, L]
    f_var_ref: AnyNDArray = W @ g_var_tmp @ W.T  # [S, N, N, P, P]
    f_var_ref = np.transpose(f_var_ref, [0, 1, 3, 2, 4])  # [S, N, P, N, P]

    if not full_cov:
        g_var_diag = np.array([g_var_diag[:, :, n, n] for n in range(N)])  # [N, L, S]
        g_var_diag = np.transpose(g_var_diag, [2, 0, 1])  # [S, N, L]

    # run gpflow's implementation
    f_mu, f_var = mix_latent_gp(
        tf.convert_to_tensor(W),
        tf.convert_to_tensor(g_mu),
        tf.convert_to_tensor(g_var_diag),
        full_cov,
        full_output_cov,
    )

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
