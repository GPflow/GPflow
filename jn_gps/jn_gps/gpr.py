from typing import Tuple

import jax.numpy as np
import jax.scipy.linalg as sl
import jax.scipy.stats as ss
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shapes
from tensorflow_probability.substrates import jax as tfp

from .clastion import Clastion, derived, put
from .clastion.integration.check_shapes import shape
from .clastion.integration.jax import arrayput
from .covariances import CovarianceFunction
from .means import MeanFunction
from .parameter import parameterput


class GPR(Clastion):
    @put()
    def jitter(self) -> float:
        return 1e-6

    mean_func = put(MeanFunction)
    covariance_func = put(CovarianceFunction)
    noise_var = parameterput(tfp.bijectors.Softplus(), shape("[]"))

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
    def n_predict(self) -> int:
        return self.n_rows(self.x_predict)

    @derived(shape("[n_data, 1]"))
    def mu_f(self) -> AnyNDArray:
        return self.mean_func(x=self.x_data).f

    @derived(shape("[n_predict, 1]"))
    def mu_x(self) -> AnyNDArray:
        return self.mean_func(x=self.x_predict).f

    @derived(shape("[n_data, n_data]"))
    def K_ff(self) -> AnyNDArray:
        return self.covariance_func(x1=self.x_data, x2=self.x_data).full + (
            self.jitter + self.noise_var.t
        ) * np.eye(self.n_data)

    @derived(shape("[n_data, n_predict]"))
    def K_fx(self) -> AnyNDArray:
        return self.covariance_func(x1=self.x_data, x2=self.x_predict).full

    @derived(shape("[n_predict, n_predict]"))
    def K_xx(self) -> AnyNDArray:
        return self.covariance_func(
            x1=self.x_predict, x2=self.x_predict
        ).full + self.jitter * np.eye(self.n_predict)

    @derived()
    def K_ff_cho_factor(self) -> Tuple[AnyNDArray, bool]:
        return sl.cho_factor(self.K_ff)  # type: ignore

    @derived(shape("[n_predict, n_data]"))
    def K_ff_inv_fx_T(self) -> AnyNDArray:
        K_ff_inv_fx = sl.cho_solve(self.K_ff_cho_factor, self.K_fx)
        return K_ff_inv_fx.T

    @derived(shape("[n_predict, 1]"))
    def f_mean(self) -> AnyNDArray:
        return self.mu_x + self.K_ff_inv_fx_T @ (self.y_data - self.mu_f)

    @derived(shape("[n_predict, 1]"))
    def y_mean(self) -> AnyNDArray:
        return self.f_mean

    @derived(shape("[n_predict, n_predict]"))
    def f_covariance(self) -> AnyNDArray:
        return self.K_xx - self.K_ff_inv_fx_T @ self.K_fx

    @derived(shape("[n_predict, n_predict]"))
    def y_covariance(self) -> AnyNDArray:
        return self.f_covariance + self.noise_var.t * np.eye(self.n_predict)

    @derived(shape("[]"))
    def log_likelihood(self) -> AnyNDArray:
        return ss.multivariate_normal.logpdf(self.y_data[:, 0], self.mu_f[:, 0], self.K_ff)
