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

from collections import namedtuple

import tensorflow as tf

from ..base import Parameter
from ..config import default_float, default_int
from ..utilities import add_noise_cov, to_default_float
from .sgpr import SGPR
from .training_mixins import RegressionData


class CGLB(SGPR):
    def __init__(
        self,
        data: RegressionData,
        *args,
        max_cg_error: float = 1.0,
        max_cg_iters: int = 100,
        restart_cg_iters: int = 40,
        v_grad_optimization: bool = False,
        **kwargs,
    ):
        super(CGLB, self).__init__(data, *args, **kwargs)
        n, b = self.data[1].shape
        self._v = Parameter(tf.zeros((b, n), dtype=default_float()), trainable=v_grad_optimization)
        self._max_cg_error = max_cg_error
        self._max_cg_iters = max_cg_iters
        self._restart_cg_iters = restart_cg_iters

    def logdet_term(self, common):
        """
        \math:`log |K + σ²I| <= log |Q + σ²I| + n * log(1 + tr(K - Q)/(σ²n))`
        """
        LB = common.LB
        AAT = common.AAT
        x, y = self.data
        num_data = to_default_float(tf.shape(y)[0])
        output_dim = to_default_float(tf.shape(y)[1])
        sigma_sq = self.likelihood.variance

        kdiag = self.kernel(x, full_cov=False)
        # t / σ²
        trace = tf.reduce_sum(kdiag) / sigma_sq - tf.reduce_sum(tf.linalg.diag_part(AAT))
        logdet_b = tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        logsigma_sq = num_data * tf.math.log(sigma_sq)

        # Correction term from Jensen's inequality
        logtrace = num_data * tf.math.log(1 + trace / num_data)
        return -output_dim * (logdet_b + 0.5 * logsigma_sq + 0.5 * logtrace)

    def quad_term(self, common) -> tf.Tensor:
        x, y = self.data
        err = y - self.mean_function(x)
        sigma_sq = self.likelihood.variance
        K = add_noise_cov(self.kernel.K(x), sigma_sq)

        A = common.A
        LB = common.LB
        preconditioner = NystromPreconditioner(A, LB, sigma_sq)
        err_t = tf.transpose(err)

        v_init = self._v
        if not v_init.trainable:
            v = cglb_conjugate_gradient(
                K,
                err_t,
                v_init,
                preconditioner,
                self._max_cg_error,
                self._max_cg_iters,
                self._restart_cg_iters,
            )
        else:
            v = v_init

        Kv = v @ K
        r = err_t - Kv
        _, error_bound = preconditioner(r)
        lb = tf.reduce_sum(v * (r + 0.5 * Kv))
        ub = lb + 0.5 * error_bound

        if not v_init.trainable:
            v_init.assign(v)

        return -ub


class NystromPreconditioner:
    """
    Preconditioner of the form \math:`(Q_ff + σ²I)⁻¹`,
    where \math:`A = σ⁻²L⁻¹Kᵤₓ` and \math:`B = AAᵀ + I = LᵦLᵦᵀ`
    """

    def __init__(self, A, LB, sigma_sq):
        self.A = A
        self.LB = LB
        self.sigma_sq = sigma_sq

    def __call__(self, v):
        sigma_sq = self.sigma_sq
        A = self.A
        LB = self.LB

        trans = tf.transpose
        trisolve = tf.linalg.triangular_solve
        matmul = tf.linalg.matmul
        sum = tf.reduce_sum

        v = trans(v)
        Av = matmul(A, v)
        LBinvAv = trisolve(LB, Av)
        LBinvtLBinvAv = trisolve(trans(LB), LBinvAv, lower=False)

        rv = v - matmul(A, LBinvtLBinvAv, transpose_a=True)
        vtrv = sum(rv * v)
        return trans(rv) / sigma_sq, vtrv / sigma_sq


def cglb_conjugate_gradient(K, b, initial, preconditioner, max_error, max_steps, restart_cg_step):
    """
    Conjugate gradient algorithm used in CGLB model.

    :param K: Matrix we want to backsolve from. Shape [N, N].
    :param b: Vector we want to backsolve. Shape [B, N].
    :param initial: Initial vector solution. Shape [N].
    :param preconditioner: Preconditioner function.
    :param max_error: Expected maximum error. This value is used as a
        decision boundary against stopping criteria.
    :param max_steps: Maximum number of CG iterations.
    :param restart_cg_step: Restart step at which the CG resets the internal state to
        the initial position using the currect solution vector \math:`v`.
    :return: Approximate solution to \math:`K v = b`.
    """
    CGState = namedtuple("CGState", ["i", "v", "r", "p", "rz"])

    def stopping_criterion(state):
        return (0.5 * state.rz > max_error) and (state.i < max_steps)

    def cg_step(state):
        Ap = state.p @ K
        denom = tf.reduce_sum(state.p * Ap, axis=-1)
        gamma = state.rz / denom
        v = state.v + gamma * state.p
        i = state.i + 1
        r = tf.cond(
            state.i % restart_cg_step == restart_cg_step - 1,
            lambda: b - v @ K,
            lambda: state.r - gamma * Ap,
        )
        z, new_rz = preconditioner(r)
        p = tf.cond(
            state.i % restart_cg_step == restart_cg_step - 1,
            lambda: z,
            lambda: z + state.p * new_rz / state.rz,
        )
        return [CGState(i, v, r, p, new_rz)]

    Kv = b @ K
    r = b - Kv
    z, rz = preconditioner(r)
    p = z
    i = tf.constant(0, dtype=default_int())
    initial_state = CGState(i, initial, r, p, rz)
    final_state = tf.while_loop(stopping_criterion, cg_step, [initial_state])
    final_state = tf.nest.map_structure(tf.stop_gradient, final_state)
    return final_state[0].v
