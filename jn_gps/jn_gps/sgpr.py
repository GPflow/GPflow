from math import tau

import jax.numpy as np
import jax.scipy.linalg as sl
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shapes
from tensorflow_probability.substrates import jax as tfp

from .clastion import Clastion, derived, put
from .clastion.integration.check_shapes import shape
from .clastion.integration.jax import arrayput
from .covariances import CovarianceFunction
from .means import MeanFunction
from .parameter import parameterput


class SGPR(Clastion):
    @put()
    def jitter(self) -> float:
        return 1e-6

    mean_func = put(MeanFunction)
    covariance_func = put(CovarianceFunction)

    noise_var = parameterput(tfp.bijectors.Softplus(), shape("[]"))

    z = parameterput(shape("[n_inducing, 1]"))

    x_data = arrayput(shape("[n_data, 1]"))
    y_data = arrayput(shape("[n_data, 1]"))
    x_predict = arrayput(shape("[n_predict, 1]"))

    @check_shapes(
        "x: [n_rows, ...]",
    )
    def n_rows(self, x: AnyNDArray) -> int:
        n_rows = x.shape[0]
        assert isinstance(n_rows, int)
        return n_rows

    @derived()
    def n_data(self) -> int:
        return self.n_rows(self.x_data)

    @derived()
    def n_inducing(self) -> int:
        return self.n_rows(self.z.t)

    @derived()
    def n_predict(self) -> int:
        return self.n_rows(self.x_predict)

    @derived()
    def n_outputs(self) -> int:
        return 1

    @derived(shape("[n_data, 1]"))
    def mu_f(self) -> AnyNDArray:
        return self.mean_func(x=self.x_data).f

    @derived(shape("[n_predict, 1]"))
    def mu_x(self) -> AnyNDArray:
        return self.mean_func(x=self.x_predict).f

    @derived(shape("[n_data, 1]"))
    def err(self) -> AnyNDArray:
        return self.y_data - self.mu_f

    @derived(shape("[]"))
    def sigma(self) -> AnyNDArray:
        return np.sqrt(self.noise_var.t)

    @derived(shape("[]"))
    def sigma_sq(self) -> AnyNDArray:
        return self.noise_var.t

    @derived(shape("[n_data]"))
    def K_f(self) -> AnyNDArray:
        # TODO(jesper): Don't compute the full covariance...
        return self.covariance_func(x1=self.x_data, x2=self.x_data).diag

    @derived(shape("[n_inducing, n_data]"))
    def K_uf(self) -> AnyNDArray:
        return self.covariance_func(x1=self.z.t, x2=self.x_data).full

    @derived(shape("[n_inducing, n_inducing]"))
    def K_uu(self) -> AnyNDArray:
        return self.covariance_func(x1=self.z.t, x2=self.z.t).full + (self.jitter) * np.eye(
            self.n_inducing
        )

    @derived(shape("[n_inducing, n_predict]"))
    def K_ux(self) -> AnyNDArray:
        return self.covariance_func(x1=self.z.t, x2=self.x_predict).full

    @derived(shape("[n_predict, n_predict]"))
    def K_xx(self) -> AnyNDArray:
        return self.covariance_func(
            x1=self.x_predict, x2=self.x_predict
        ).full + self.jitter * np.eye(self.n_predict)

    @derived(shape("[n_inducing, n_inducing]"))
    def L(self) -> AnyNDArray:
        return sl.cholesky(self.K_uu, lower=True)

    @derived(shape("[n_inducing, n_data]"))
    def A(self) -> AnyNDArray:
        return sl.solve_triangular(self.L, self.K_uf, lower=True) / self.sigma
        # return sl.solve_triangular(self.L, self.K_uf / self.sigma, lower=True)

    @derived(shape("[n_inducing, n_inducing]"))
    def AA_T(self) -> AnyNDArray:
        # TODO(jesper): Avoid the transpose. einsum?
        return self.A @ self.A.T

    @derived(shape("[n_inducing, n_inducing]"))
    def B(self) -> AnyNDArray:
        return np.eye(self.n_inducing) + self.AA_T

    @derived(shape("[n_inducing, n_inducing]"))
    def L_B(self) -> AnyNDArray:
        return sl.cholesky(self.B, lower=True)

    @derived(shape("[n_inducing, 1]"))
    def c(self) -> AnyNDArray:
        return sl.solve_triangular(self.L_B, self.A @ self.err, lower=True) / self.sigma
        # return sl.solve_triangular(self.L_B, self.A @ self.err / self.sigma, lower=True)

    @derived(shape("[]"))
    def log_likelihood(self) -> AnyNDArray:
        t1 = -0.5 * self.n_data * np.log(tau)
        # t2 = -0.5 * np.log(np.linalg.det(self.B))
        t2 = -self.n_outputs * np.sum(np.log(np.diag(self.L_B)))
        t3 = -0.5 * self.n_outputs * self.n_data * np.log(self.sigma_sq)
        t4 = -0.5 * np.sum(self.err ** 2) / self.sigma_sq  # err.T @ err / sigma_sq
        t5 = 0.5 * np.sum(self.c ** 2)  # c.T @ c
        t6 = -0.5 * self.n_outputs * np.sum(self.K_f) / self.sigma_sq
        t7 = 0.5 * self.n_outputs * np.trace(self.AA_T)

        return t1 + t2 + t3 + t4 + t5 + t6 + t7

    @derived(shape("[n_inducing, n_predict]"))
    def tmp1(self) -> AnyNDArray:
        return sl.solve_triangular(self.L, self.K_ux, lower=True)

    @derived(shape("[n_inducing, n_predict]"))
    def tmp2(self) -> AnyNDArray:
        return sl.solve_triangular(self.L_B, self.tmp1, lower=True)

    @derived(shape("[n_predict, 1]"))
    def f_mean(self) -> AnyNDArray:
        return self.tmp2.T @ self.c + self.mu_x

    @derived(shape("[n_predict, n_predict]"))
    def f_covariance(self) -> AnyNDArray:
        return self.K_xx + self.tmp2.T @ self.tmp2 - self.tmp1.T @ self.tmp1

    @derived(shape("[n_predict, 1]"))
    def y_mean(self) -> AnyNDArray:
        return self.f_mean

    @derived(shape("[n_predict, n_predict]"))
    def y_covariance(self) -> AnyNDArray:
        return self.f_covariance + self.sigma_sq * np.eye(self.n_predict)
