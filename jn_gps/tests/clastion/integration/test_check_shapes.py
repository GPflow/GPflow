import jax.numpy as np
import pytest
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes.exceptions import ShapeMismatchError

from jn_gps.clastion import Clastion, derived
from jn_gps.clastion.integration.check_shapes import shape
from jn_gps.clastion.integration.jax import arrayput


def test_shape() -> None:
    class Foo(Clastion):
        a = arrayput(shape("[1, m]"))
        b = arrayput(shape("[n, 1]"))
        c = arrayput(shape("[n, m]"))

        @derived(shape("[n, m]"))
        def ab(self) -> AnyNDArray:
            return self.a + self.b

    foo = Foo(a=np.zeros((1, 3)), b=np.ones((4, 1)))
    foo(c=np.ones((4, 3)))
    foo.ab  # pylint: disable=pointless-statement

    with pytest.raises(ShapeMismatchError):
        foo(c=np.ones((3, 4)))

    with pytest.raises(ShapeMismatchError):
        Foo(a=np.zeros((1, 3)), b=np.ones((4, 2)))
