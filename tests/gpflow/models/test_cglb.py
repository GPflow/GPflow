# Copyright 2021 the GPflow authors.
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

from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from gpflow.config import default_float
from gpflow.kernels import SquaredExponential
from gpflow.models import CGLB, SGPR
from gpflow.models.cglb import NystromPreconditioner, cglb_conjugate_gradient
from gpflow.utilities import to_default_float as tdf


def data(rng: np.random.RandomState):
    n: int = 100
    d: int = 2

    x = rng.randn(n, d)
    c = np.array([[-1.4], [0.5]])
    y = np.sin(x @ c + 0.5 * rng.randn(n, 1))
    z = rng.randn(10, 2)

    return (tdf(x), tdf(y)), tdf(z)


def test_cglb_check_basics():
    """
    * Quadratic term of the CGLB with v=0 equivalent to the quadratic term of the SGPR.
    * Log determinant term of the CGLB is less or equal to SGPR log determinant.
    """

    rng: np.random.RandomState = np.random.RandomState(999)
    train, z = data(rng)
    noise = 0.2

    sgpr = SGPR(train, kernel=SquaredExponential(), inducing_variable=z, noise_variance=noise)

    # `v_grad_optimization=True` turns off the CG in the quadratic term
    cglb = CGLB(
        train,
        kernel=SquaredExponential(),
        inducing_variable=z,
        noise_variance=noise,
        v_grad_optimization=True,
    )

    sgpr_common = sgpr._common_calculation()
    cglb_common = cglb._common_calculation()

    sgpr_quad_term = sgpr.quad_term(sgpr_common)
    cglb_quad_term = cglb.quad_term(cglb_common)
    np.testing.assert_almost_equal(sgpr_quad_term, cglb_quad_term)

    sgpr_logdet = sgpr.logdet_term(sgpr_common)
    cglb_logdet = cglb.logdet_term(cglb_common)
    assert cglb_logdet >= sgpr_logdet

    x = train[0]
    K = SquaredExponential()(x) + noise * tf.eye(x.shape[0], dtype=default_float())
    gpr_logdet = -0.5 * tf.linalg.logdet(K)
    assert cglb_logdet <= gpr_logdet


def test_conjugate_gradient_convergence():
    """
    Check that the method of conjugate gradients implemented can solve a linear system of equations
    """
    rng: np.random.RandomState = np.random.RandomState(999)
    noise = 1e-3
    train, z = data(rng)
    x, y = train
    n = x.shape[0]
    b = tf.transpose(y)
    k = SquaredExponential()
    K = k(x) + noise * tf.eye(n, dtype=default_float())
    Kinv_y = tf.linalg.solve(K, y)  # We could solve by cholesky instead

    model = CGLB((x, y), kernel=k, inducing_variable=z, noise_variance=noise)
    common = model._common_calculation()

    initial = tf.zeros_like(b)
    A = common.A
    LB = common.LB
    max_error = 0.1
    max_steps = 200
    restart_cg_step = 200
    preconditioner = NystromPreconditioner(A, LB, noise)

    v = cglb_conjugate_gradient(
        K, b, initial, preconditioner, max_error, max_steps, restart_cg_step
    )

    np.testing.assert_allclose(Kinv_y, tf.transpose(v))


def test_cglb_quad_term_guarantees():
    """
    Check that when conjugate gradient is used to evaluate the quadratic term,
    the obtained solution is:

    1. Smaller than the solution computed by Cholesky decomposition
    2. Within the error tolerance of the solution computed by Cholesky
    """
    rng: np.random.RandomState = np.random.RandomState(999)

    max_error: float = 1e-2
    noise: float = 1e-2
    train, z = data(rng)
    x, y = train
    k = SquaredExponential()
    K = k(x) + noise * tf.eye(x.shape[0], dtype=default_float())

    def inv_quad_term(K: tf.Tensor, y: tf.Tensor):
        """
        For PSD K, compute -0.5 * y.T K^{-1} y via Cholesky decomposition
        """
        L = tf.linalg.cholesky(K)
        Linvy = tf.linalg.triangular_solve(L, y)
        return -0.5 * tf.reduce_sum(tf.square(Linvy))

    cholesky_quad_term = inv_quad_term(K, y)

    cglb = CGLB(
        train,
        kernel=k,
        inducing_variable=z,
        noise_variance=noise,
        max_cg_error=max_error,
        max_cg_iters=100,
        restart_cg_iters=10,
    )

    common = cglb._common_calculation()
    cglb_quad_term = cglb.quad_term(common)

    assert cglb_quad_term <= cholesky_quad_term
    assert cglb_quad_term >= cholesky_quad_term - max_error
