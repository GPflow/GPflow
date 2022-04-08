from typing import Callable

from gpflow.base import AnyNDArray
from jax import random

from .clastion import Clastion, Put, derived, put
from .clastion.integration.check_shapes import shape


class XYData(Clastion):
    @derived(shape("[n_rows, n_inputs]"))
    def x_data(self) -> AnyNDArray:
        ...

    @derived(shape("[n_rows, n_outputs]"))
    def y_data(self) -> AnyNDArray:
        ...

    @derived()
    def n_rows(self) -> int:
        return self.x_data.shape[0]  # type: ignore[no-any-return]

    @derived()
    def n_inputs(self) -> int:
        return self.x_data.shape[1]  # type: ignore[no-any-return]

    @derived()
    def n_outputs(self) -> int:
        return self.y_data.shape[1]  # type: ignore[no-any-return]


class NoisyFnXYData(XYData):

    key: Put[random.KeyArray] = put()

    @derived()
    def _split_key(self) -> random.KeyArray:
        return random.split(self.key, 2)

    @derived()
    def _x_key(self) -> random.KeyArray:
        return self._split_key[0]

    @derived()
    def _noise_key(self) -> random.KeyArray:
        return self._split_key[1]

    @put()
    def n_rows(self) -> int:
        return 50

    @put()
    def n_inputs(self) -> int:
        return 1

    @put()
    def noise_scale(self) -> float:
        return 1.0

    @put(shape("[n_rows, n_inputs]"))
    def x_data(self) -> AnyNDArray:
        return random.uniform(self._x_key, (self.n_rows, self.n_inputs))

    f: Put[Callable[[AnyNDArray], AnyNDArray]] = put()

    @derived(shape("[n_rows, n_outputs]"))
    def y_data(self) -> AnyNDArray:
        f = self.f(self.x_data)
        noise = random.normal(self._noise_key, f.shape)
        return f + self.noise_scale * noise
