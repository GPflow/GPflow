from typing import Any

import jax.numpy as np
from gpflow.base import AnyNDArray
from tensorflow_probability.substrates import jax as tfp

from .clastion import Clastion, derived
from .clastion.integration.check_shapes import shape
from .clastion.integration.jax import arrayput
from .parameter import parameterput


class CovarianceFunction(Clastion):

    x1 = arrayput(shape("[n1, 1]"))
    x2 = arrayput(shape("[n2, 1]"))

    @derived(shape("[n1]"))
    def diag(self) -> AnyNDArray:
        raise NotImplementedError

    @derived(shape("[n1, n2]"))
    def full(self) -> AnyNDArray:
        raise NotImplementedError


class RBF(CovarianceFunction):
    """
    Radial Basis Function (RBF) kernel.

    Also known as a Squared Exponential (SE) kernel.
    """

    @parameterput(tfp.bijectors.Softplus(), shape("[]"))
    def variance(self) -> Any:
        return 1.0

    @parameterput(tfp.bijectors.Softplus(), shape("[]"))
    def lengthscale(self) -> Any:
        return 1.0

    @derived(shape("[n1]"))
    def diag(self) -> AnyNDArray:
        return np.full(self.x1.shape[:-1], self.variance.t)

    @derived(shape("[n1, n2]"))
    def full(self) -> AnyNDArray:
        errs = self.x1[:, None, :] - self.x2[None, :, :]
        errs = errs / self.lengthscale.t
        sum_sq_errs = np.sum(errs ** 2, axis=-1)
        return self.variance.t * np.exp(-0.5 * sum_sq_errs)
