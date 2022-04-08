# fmt: off
import jax

jax.config.update("jax_enable_x64", True)  # type: ignore[no-untyped-call]
# fmt: on

from typing import Tuple

from gpflow.experimental.check_shapes import ErrorContext, Shape, get_shape
from jax.core import Tracer
from jaxlib.xla_extension import DeviceArray  # pylint: disable=no-name-in-module

__version__ = "0.1.0"


@get_shape.register(DeviceArray)
def get_jax_array_shape(shaped: DeviceArray, context: ErrorContext) -> Shape:
    result: Tuple[int, ...] = shaped.shape
    return result


@get_shape.register(Tracer)
def get_jax_tracer_shape(shaped: Tracer, context: ErrorContext) -> Shape:
    result: Tuple[int, ...] = shaped.shape
    return result


__no_all__ = None
