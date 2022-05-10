import copy
import pickle
from typing import Any, Type

import pytest
import tensorflow as tf
import tensorflow_probability as tfp
from packaging.version import Version

import gpflow
from gpflow.base import TensorType
from gpflow.experimental.check_shapes import check_shapes


class A(tf.Module):
    def __init__(self) -> None:
        self.var = tf.Variable([1.0])
        shift = tf.Variable(1e-6)
        self.bijector = tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(shift)])

    @check_shapes(
        "x: [any...]",
        "return: [any...]",
    )
    def __call__(self, x: TensorType) -> tf.Tensor:
        return self.bijector(x)


class B(tf.Module):
    def __init__(self) -> None:
        self.var = tf.Variable([2.0])
        self.a = A()

    @check_shapes(
        "x: [any...]",
        "return: [any...]",
    )
    def __call__(self, x: TensorType) -> tf.Tensor:
        return self.a(x)


class C(tf.Module):
    def __init__(self) -> None:
        self.var = gpflow.Parameter([2.0], transform=gpflow.utilities.positive(lower=1e-6))


class D(tf.Module):
    def __init__(self) -> None:
        self.var = gpflow.Parameter([10.0], transform=gpflow.utilities.positive())
        self.var2 = gpflow.Parameter([5.0])
        self.c = C()


class NestedModule(tf.Module):
    def __init__(self, module: tf.Module) -> None:
        self.module = module


@pytest.mark.parametrize("module", [A(), B()])
def test_clears_bijector_cache_and_deepcopy(module: Type[Any]) -> None:
    """
    With each forward pass through a bijector, a cache is stored inside which prohibits the deepcopy of the bijector.
    This is due to the fact that HashableWeakRef objects are not pickle-able, which raises a TypeError. Alternatively,
    one can make use of `deepcopy_component` to deepcopy a module containing used bijectors.
    """
    input = 1.0
    _ = module(input)

    if Version(tfp.__version__) >= Version("0.12"):
        module_copy = copy.deepcopy(module)
    else:
        # older versions of tensorflow_probability required us to maintain a workaround
        with pytest.raises(TypeError):
            copy.deepcopy(module)
        module_copy = gpflow.utilities.deepcopy(module)

    assert module.var == module_copy.var
    assert module.var is not module_copy.var
    module_copy.var.assign([5.0])
    assert module.var != module_copy.var


def test_freeze() -> None:
    module = NestedModule(NestedModule(A()))
    module_frozen = gpflow.utilities.freeze(module)
    assert len(module.variables) == 2
    assert module_frozen.variables == ()
    assert isinstance(module.module.module.var, tf.Variable)
    assert isinstance(module_frozen.module.module.var, tf.Tensor)


def test_pickle_frozen() -> None:
    """
    Regression test for the bug described in GPflow/GPflow#1338
    """
    module = D()
    module_frozen = gpflow.utilities.freeze(module)

    pickled = pickle.dumps(module_frozen)
    loaded = pickle.loads(pickled)

    assert loaded.var == module_frozen.var
    assert loaded.var2 == module_frozen.var2
    assert loaded.c.var == module_frozen.c.var
