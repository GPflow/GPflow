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

from typing import Tuple, cast

import numpy as np
import tensorflow as tf

from gpflow.base import AnyNDArray, RegressionData
from gpflow.config import default_float
from gpflow.experimental.check_shapes import check_shapes
from gpflow.kernels import SquaredExponential
from gpflow.models import CGLB, GPR, SGPR
from gpflow.models.cglb import NystromPreconditioner, cglb_conjugate_gradient
from gpflow.utilities import to_default_float as tdf


@check_shapes(
    "return[0][0]: [N, D]  # X",
    "return[0][1]: [N, P]  # Y",
    "return[1]: [M, D]  # Z, inducing points",
    "return[2]: [Nnew, D]  # Xnew",
)
def data(rng: np.random.RandomState) -> Tuple[RegressionData, tf.Tensor, tf.Tensor]:
    n: int = 100
    t: int = 20
    d: int = 2

    x: AnyNDArray = rng.randn(n, d)
    xs: AnyNDArray = rng.randn(t, d)  # test points
    c: AnyNDArray = np.array([[-1.4], [0.5]])
    y = np.sin(cast(AnyNDArray, x @ c) + 0.5 * rng.randn(n, 1))
    z = rng.randn(10, 2)

    return (tdf(x), tdf(y)), tdf(z), tdf(xs)


def test_cglb_check_basics() -> None:
    """
    * Quadratic term of CGLB with v=0 is equivalent to the quadratic term of SGPR.
    * Log determinant term of CGLB is less or equal to SGPR log determinant.
        In the test the `logdet_term` method returns negative half of the logdet bound,
        therefore we run the opposite direction of the sign.
    """

    rng: np.random.RandomState = np.random.RandomState(999)
    train, z, _ = data(rng)
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


def test_conjugate_gradient_convergence() -> None:
    """
    Check that the method of conjugate gradients implemented can solve a linear system of equations
    """
    rng: np.random.RandomState = np.random.RandomState(999)
    noise = 1e-3
    train, z, _ = data(rng)
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
    max_error = 0.01
    max_steps = 200
    restart_cg_step = 200
    preconditioner = NystromPreconditioner(A, LB, noise)

    v = cglb_conjugate_gradient(
        K, b, initial, preconditioner, max_error, max_steps, restart_cg_step
    )

    # NOTE: with smaller `max_error` we can reduce the `rtol`
    np.testing.assert_allclose(Kinv_y, tf.transpose(v), rtol=0.1)


def test_cglb_quad_term_guarantees() -> None:
    """
    Check that when conjugate gradient is used to evaluate the quadratic term,
    the obtained solution is:

    1. Smaller than the solution computed by Cholesky decomposition
    2. Within the error tolerance of the solution computed by Cholesky
    """
    rng: np.random.RandomState = np.random.RandomState(999)

    max_error: float = 1e-2
    noise: float = 1e-2
    train, z, _ = data(rng)
    x, y = train
    k = SquaredExponential()
    K = k(x) + noise * tf.eye(x.shape[0], dtype=default_float())

    def inv_quad_term(K: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
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
        cg_tolerance=max_error,
        max_cg_iters=100,
        restart_cg_iters=10,
    )

    common = cglb._common_calculation()
    cglb_quad_term = cglb.quad_term(common)

    assert cglb_quad_term <= cholesky_quad_term
    assert np.abs(cglb_quad_term - cholesky_quad_term) <= max_error


def test_cglb_predict() -> None:
    """
    Test that 1.) The predict method returns the same variance estimate as SGPR.
              2.) The predict method returns the same mean as SGPR for v=0.
              3.) The predict method returns a mean very similar to GPR when CG is run to low tolerance.
    """
    rng: np.random.RandomState = np.random.RandomState(999)
    train, z, xs = data(rng)
    noise = 0.2

    gpr = GPR(train, kernel=SquaredExponential(), noise_variance=noise)
    sgpr = SGPR(train, kernel=SquaredExponential(), inducing_variable=z, noise_variance=noise)

    cglb = CGLB(
        train,
        kernel=SquaredExponential(),
        inducing_variable=z,
        noise_variance=noise,
    )

    gpr_mean, _ = gpr.predict_y(xs, full_cov=False)
    sgpr_mean, sgpr_cov = sgpr.predict_y(xs, full_cov=False)
    cglb_mean, cglb_cov = cglb.predict_y(
        xs, full_cov=False, cg_tolerance=1e6
    )  # set tolerance high so v stays at 0.

    assert np.allclose(sgpr_cov, cglb_cov)
    assert np.allclose(sgpr_mean, cglb_mean)

    cglb_mean, _ = cglb.predict_y(xs, full_cov=False, cg_tolerance=1e-12)

    assert np.allclose(gpr_mean, cglb_mean)
