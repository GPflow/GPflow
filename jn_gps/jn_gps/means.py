import jax.numpy as np
from gpflow.base import AnyNDArray

from .clastion import Clastion, derived
from .clastion.integration.check_shapes import shape
from .clastion.integration.jax import arrayput
from .parameter import parameterput


class MeanFunction(Clastion):

    x = arrayput(shape("[n_rows, n_inputs]"))

    @derived(shape("[n_rows, 1]"))
    def f(self) -> AnyNDArray:
        raise NotImplementedError


class ZeroMeanFunction(MeanFunction):
    @derived(shape("[n_rows, 1]"))
    def f(self) -> AnyNDArray:
        return np.zeros_like(self.x)


class PolynomialMeanFunction(MeanFunction):

    coeffs = parameterput(shape("[degrees...]"))

    @derived(shape("[n_rows, 1]"))
    def f(self) -> AnyNDArray:
        x = self.x
        coeffs = self.coeffs.t
        rank = len(coeffs.shape)
        assert x.shape[-1] == rank
        powers = np.asarray(1)
        for i in range(rank):
            dim_size = coeffs.shape[i]
            p = x[:, i, None] ** np.arange(dim_size)[None, :]
            new_shape = [-1] + i * [1] + [dim_size] + (rank - i - 1) * [1]
            p = np.reshape(p, new_shape)
            powers *= p
        result = coeffs * powers
        for i in range(rank):
            result = np.sum(result, axis=1)
        return result[:, None]
