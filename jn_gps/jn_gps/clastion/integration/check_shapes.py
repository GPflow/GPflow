from typing import Any

from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.experimental.check_shapes.error_contexts import (
    AttributeContext,
    ObjectTypeContext,
    StackContext,
)

from ..clastion import Clastion, Preprocessor, Put


class shape(Preprocessor):
    def __init__(self, spec: str) -> None:
        self._spec = spec

    def process(self, instance: Clastion, key: Put[Any], value: Any) -> Any:
        # pylint: disable=protected-access
        assert key._name
        shape_checker = getattr(instance, "__shape_checker__", None)
        if not shape_checker:
            shape_checker = ShapeChecker()
            setattr(instance, "__shape_checker__", shape_checker)
        shape_checker.check_shape(
            value,
            self._spec,
            StackContext(ObjectTypeContext(instance), AttributeContext(key._name)),
        )
        return value
