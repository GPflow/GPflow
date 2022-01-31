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
from typing import NamedTuple, Union

import tensorflow as tf

from ..base import InputData, MeanAndVariance, Parameter, RegressionData
from ..config import default_float, default_int
from ..covariances import Kuf
from ..utilities import add_noise_cov, to_default_float
from .sgpr import SGPR


class CGLB(SGPR):
    """
    Conjugate Gradient Lower Bound. The key reference is

    ::

        @InProceedings{pmlr-v139-artemev21a,
            title = {Tighter Bounds on the Log Marginal Likelihood of
            Gaussian Process Regression Using Conjugate Gradients},
            author = {Artemev, Artem and Burt, David R. and van der Wilk, Mark},
            booktitle = {Proceedings of the 38th International Conference on Machine Learning},
            pages = {362--372},
            year = {2021}
        }

    :param cg_tolerance: Determines accuracy to which conjugate
        gradient is run when evaluating the elbo. Running more
        iterations of CG would increase the ELBO by at most
        `cg_tolerance`.

    :param max_cg_iters: Maximum number of iterations of CG to run
        per evaluation of the ELBO (or mean prediction).

    :param restart_cg_iters: How frequently to restart the CG iteration.
        Can be useful to avoid build up of numerical errors when
        many steps of CG are run.

    :param v_grad_optimization: If False, in every evaluation of the
        ELBO, CG is run to select a new auxilary vector `v`. If
        False, no CG is run when evaluating the ELBO but
        gradients with respect to `v` are tracked so that it can
        be optimized jointly with other parameters.

    """

    def __init__(
        self,
        data: RegressionData,
        *args,
        cg_tolerance: float = 1.0,
        max_cg_iters: int = 100,
        restart_cg_iters: int = 40,
        v_grad_optimization: bool = False,
        **kwargs,
    ):
        super(CGLB, self).__init__(data, *args, **kwargs)
        n, b = self.data[1].shape
        self._v = Parameter(tf.zeros((b, n), dtype=default_float()), trainable=v_grad_optimization)
        self._cg_tolerance = cg_tolerance
        self._max_cg_iters = max_cg_iters
        self._restart_cg_iters = restart_cg_iters

    @property
    def aux_vec(self):
        return self._v

    def logdet_term(self, common: NamedTuple) -> tf.Tensor:
        """
        Compute a lower bound on -0.5 * log |K + σ²I| based on a
        low-rank approximation to K.
        ..  math::
            log |K + σ²I| <= log |Q + σ²I| + n * log(1 + tr(K - Q)/(σ²n)).

        This bound is at least as tight as
        ..  math::
            log |K + σ²I| <=  log |Q + σ²I| + tr(K - Q)/σ²,

        which appears in SGPR.
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

    def quad_term(self, common: NamedTuple) -> tf.Tensor:
        """
        Computes a lower bound on the quadratic term in the log
        marginal likelihood of conjugate GPR. The bound is based on
        an auxiliary vector, v. For :math:`Q ≺ K` and :math:`r=y - Kv`

        .. math::
            -0.5 * (rᵀQ⁻¹r + 2yᵀv - vᵀ K v ) <= -0.5 * yᵀK⁻¹y <= -0.5 * (2yᵀv - vᵀKv).

        Equality holds if :math:`r=0`, i.e. :math:`v = K⁻¹y`.

        If `self.aux_vec` is trainable, gradients are computed with
        respect to :math:`v` as well and :math:`v` can be optimized
        using gradient based methods.

        Otherwise, :math:`v` is updated with the method of conjugate
        gradients (CG). CG is run until :math:`0.5 * rᵀQ⁻¹r <= ϵ`,
        which ensures that the maximum bias due to this term is not
        more than :math:`ϵ`. The :math:`ϵ` is the CG tolerance.
        """
        x, y = self.data
        err = y - self.mean_function(x)
        sigma_sq = self.likelihood.variance
        K = add_noise_cov(self.kernel.K(x), sigma_sq)

        A = common.A
        LB = common.LB
        preconditioner = NystromPreconditioner(A, LB, sigma_sq)
        err_t = tf.transpose(err)

        v_init = self.aux_vec
        if not v_init.trainable:
            v = cglb_conjugate_gradient(
                K,
                err_t,
                v_init,
                preconditioner,
                self._cg_tolerance,
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

    def predict_f(
        self,
        xnew: InputData,
        full_cov=False,
        full_output_cov=False,
        cg_tolerance: Union[float, None] = 1e-3,
    ) -> MeanAndVariance:
        """
        The posterior mean for CGLB model is given by

        .. :math::
            m(xs) = K_{sf}v + Q_{ff}Q⁻¹r

        where :math:`r = y - K v`  is the residual from CG.

        Note that when :math:`v=0`, this agree with the SGPR mean,
        while if :math:`v = K⁻¹ y`, then :math:`r=0`, and the exact
        GP mean is recovered.

        :param cg_tolerance: float or None: If None, the cached value of
            :math:`v` is used. If float, conjugate gradient is run until :math:`rᵀQ⁻¹r < ϵ`.
        """
        x, y = self.data
        err = y - self.mean_function(x)
        kxx = self.kernel(x, x)
        ksf = self.kernel(xnew, x)
        sigma_sq = self.likelihood.variance
        sigma = tf.sqrt(sigma_sq)
        iv = self.inducing_variable
        kernel = self.kernel
        matmul = tf.linalg.matmul
        trisolve = tf.linalg.triangular_solve

        kmat = add_noise_cov(kxx, sigma_sq)

        common = self._common_calculation()
        A, LB, L = common.A, common.LB, common.L

        v = self.aux_vec
        if cg_tolerance is not None:
            preconditioner = NystromPreconditioner(A, LB, sigma_sq)
            err_t = tf.transpose(err)
            v = cglb_conjugate_gradient(
                kmat,
                err_t,
                v,
                preconditioner,
                cg_tolerance,
                self._max_cg_iters,
                self._restart_cg_iters,
            )

        cg_mean = matmul(ksf, v, transpose_b=True)
        res = err - matmul(kmat, v, transpose_b=True)

        Kus = Kuf(iv, kernel, xnew)
        Ares = matmul(A, res)  # The god of war!
        c = trisolve(LB, Ares, lower=True) / sigma
        tmp1 = trisolve(L, Kus, lower=True)
        tmp2 = trisolve(LB, tmp1, lower=True)
        sgpr_mean = matmul(tmp2, c, transpose_a=True)

        if full_cov:
            var = (
                kernel(xnew)
                + matmul(tmp2, tmp2, transpose_a=True)
                - matmul(tmp1, tmp1, transpose_a=True)
            )
            var = tf.tile(var[None, ...], [self.num_latent_gps, 1, 1])  # [P, N, N]
        else:
            var = (
                kernel(xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), 0)
                - tf.reduce_sum(tf.square(tmp1), 0)
            )
            var = tf.tile(var[:, None], [1, self.num_latent_gps])

        mean = sgpr_mean + cg_mean + self.mean_function(xnew)
        return mean, var

    def predict_y(
        self,
        xnew: InputData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        cg_tolerance: Union[float, None] = 1e-3,
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the held-out data at the
        input points.
        """
        if full_cov or full_output_cov:
            # See https://github.com/GPflow/GPflow/issues/1461
            raise NotImplementedError(
                "The predict_y method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        f_mean, f_var = self.predict_f(
            xnew, full_cov=full_cov, full_output_cov=full_output_cov, cg_tolerance=cg_tolerance
        )
        return self.likelihood.predict_mean_and_var(f_mean, f_var)

    def predict_log_density(
        self,
        data: RegressionData,
        full_cov: bool = False,
        full_output_cov: bool = False,
        cg_tolerance: Union[float, None] = 1e-3,
    ) -> tf.Tensor:
        """
        Compute the log density of the data at the new data points.
        """
        if full_cov or full_output_cov:
            raise NotImplementedError(
                "The predict_log_density method currently supports only the argument values full_cov=False and full_output_cov=False"
            )

        x, y = data
        f_mean, f_var = self.predict_f(
            x, full_cov=full_cov, full_output_cov=full_output_cov, cg_tolerance=cg_tolerance
        )
        return self.likelihood.predict_log_density(f_mean, f_var, y)


class NystromPreconditioner:
    """
    Preconditioner of the form :math:`Q=(Q_ff + σ²I)⁻¹`,
    where L is lower triangular with :math: `LLᵀ = Kᵤᵤ`
    :math:`A = σ⁻²L⁻¹Kᵤₓ` and :math:`B = AAᵀ + I = LᵦLᵦᵀ`
    """

    def __init__(self, A: tf.Tensor, LB: tf.Tensor, sigma_sq: float):
        self.A = A
        self.LB = LB
        self.sigma_sq = sigma_sq

    def __call__(self, v):
        """
        Computes :math:`vᵀQ^{-1}` and `vᵀQ^{-1}v`. Note that this is
        implemented as multipication of a row vector on the right.

        :param v: Vector we want to backsolve. Shape [B, N].
        """
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


def cglb_conjugate_gradient(
    K, b, initial, preconditioner, cg_tolerance, max_steps, restart_cg_step
):
    """
    Conjugate gradient algorithm used in CGLB model. The method of
    conjugate gradient (Hestenes and Stiefel, 1952) produces a
    sequence of vectors :math:`v_0, v_1, v_2, ..., v_N` such that
    :math:`v_0` = initial, and (in exact arithmetic)
    :math:`Kv_n = b`. In practice, the v_i often converge quickly to
    approximate :math:`K^{-1}b`, and the algorithm can be stopped
    without running N iterations.

    We assume the preconditioner, :math:`Q`, satisfies :math:`Q ≺ K`,
    and stop the algorithm when :math:`r_i = b - Kv_i` satisfies
    :math:`||rᵢᵀ||_{Q⁻¹r}^2 = rᵢᵀQ⁻¹rᵢ <= ϵ`.

    :param K: Matrix we want to backsolve from. Must be PSD. Shape [N, N].
    :param b: Vector we want to backsolve. Shape [B, N].
    :param initial: Initial vector solution. Shape [N].
    :param preconditioner: Preconditioner function.
    :param cg_tolerance: Expected maximum error. This value is used
        as a decision boundary against stopping criteria.
    :param max_steps: Maximum number of CG iterations.
    :param restart_cg_step: Restart step at which the CG resets the
        internal state to the initial position using the currect
        solution vector :math:`v`. Can help avoid build up of
        numerical errors.

    :return: `v` where `v` approximately satisfies :math:`Kv = b`.
    """
    CGState = namedtuple("CGState", ["i", "v", "r", "p", "rz"])

    def stopping_criterion(state):
        return (0.5 * state.rz > cg_tolerance) and (state.i < max_steps)

    def cg_step(state: CGState):
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

    Kv = initial @ K
    r = b - Kv
    z, rz = preconditioner(r)
    p = z
    i = tf.constant(0, dtype=default_int())
    initial_state = CGState(i, initial, r, p, rz)
    final_state = tf.while_loop(stopping_criterion, cg_step, [initial_state])
    final_state = tf.nest.map_structure(tf.stop_gradient, final_state)
    return final_state[0].v
