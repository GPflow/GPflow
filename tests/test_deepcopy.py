import pytest
import copy

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import deepcopy


class A(tf.Module):
    def __init__(self):
        self.var = tf.Variable([1.0])
        shift = tf.Variable(1e-6)
        self.bijector = tfp.bijectors.Chain([tfp.bijectors.Softplus(), tfp.bijectors.Shift(shift)])

    def __call__(self, x):
        return self.bijector(x)


class B(tf.Module):
    def __init__(self):
        self.var = tf.Variable([2.0])
        self.a = A()

    def __call__(self, x):
        return self.a(x)


class Nested(tf.Module):
    def __init__(self, module: tf.Module):
        self.module = module


@pytest.mark.parametrize("module", [A(), B()])
def test_deepcopy_clears_bijector_cache_and_deecopy(module):
    """
    With each forward pass through a bijector, a cache is stored inside which prohibits the deepcopy of the bijector.
    This is due to the fact that HashableWeakRef objects are not pickle-able, which raises a TypeError. Alternatively,
    one can make use of `deepcopy_component` to deepcopy a module containing used bijectors.
    """
    input = 1.0
    _ = module(input)
    with pytest.raises(TypeError):
        copy.deepcopy(module)
    module_copy = deepcopy(module)
    assert module.var == module_copy.var
    assert module.var is not module_copy.var
    module_copy.var.assign([5.0])
    assert module.var != module_copy.var


def test_deepcopy_with_freeze():
    module = Nested(Nested(A()))
    module_frozen = deepcopy(module, freeze=True)
    assert len(module.variables) == 2
    assert module_frozen.variables == ()
    assert isinstance(module_frozen.module.module.var, tf.Tensor)
    assert isinstance(module_frozen.module.module.var, tf.Tensor)
