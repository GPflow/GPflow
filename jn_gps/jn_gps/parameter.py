from typing import Any, List, Optional, Union

import jax.numpy as np
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import ErrorContext, Shape, get_shape
from tensorflow_probability.substrates import jax as tfp

from .clastion import Clastion, InPut, Preprocessor, Put, derived, put
from .clastion.integration.jax import arrayput


class Parameter(Clastion):

    u = arrayput()
    """
    Untransformed value.
    """

    @put()
    def bijector(self) -> tfp.bijectors.Bijector:
        return tfp.bijectors.Identity()

    @derived()
    def t(self) -> AnyNDArray:
        """
        Transformed value.
        """
        return self.bijector.forward(self.u)

    def from_transformed(self, initial_value: AnyNDArray) -> "Parameter":
        return self(u=self.bijector.inverse(initial_value))


@get_shape.register(Parameter)
def get_parameter_shape(shaped: Parameter, context: ErrorContext) -> Shape:
    return get_shape(shaped.t, context)


class asparameter(Preprocessor):
    def __init__(self, default_bijector: Optional[tfp.bijectors.Bijector]) -> None:
        self._default_bijector = default_bijector

    def process(self, instance: Clastion, key: Put[Any], value: Any) -> Any:
        if isinstance(value, Parameter):
            return value
        value = np.asarray(value)
        param = Parameter()
        if self._default_bijector:
            param = param(bijector=self._default_bijector)
        param = param.from_transformed(value)
        return param


def parameterput(
    *bijector_and_preprocessors: Union[tfp.bijectors.Bijector, Preprocessor]
) -> InPut[Parameter]:
    bijector: Optional[tfp.bijectors.Bijector] = None
    preprocessors: List[Preprocessor] = []
    for b_or_p in bijector_and_preprocessors:
        if isinstance(b_or_p, tfp.bijectors.Bijector):
            assert bijector is None
            bijector = b_or_p
        else:
            assert isinstance(b_or_p, Preprocessor)

    return put(asparameter(bijector), *preprocessors)
