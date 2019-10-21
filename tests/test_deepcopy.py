from copy import deepcopy
import pytest

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import deepcopy_components



class A(tf.Module):
    def __init__(self):
        self.var = tf.Variable([1.0])
        self.bijector = tfp.bijectors.Softplus()

    def __call__(self, x):
        return self.bijector(x)


class B(tf.Module):
    def __init__(self):
        self.var = tf.Variable([2.0])
        self.a = A()

    def __call__(self, x):
        return self.a(x)


@pytest.mark.parametrize('module', [A(), B()])
def test_deepcopy_component_clears_bijector_cache_and_deecopy(module):
    """
    With each forward pass through a bijector, a cache is stored inside which prohibits the deepcopy of the bijector.
    This is due to the fact that HashableWeakRef objects are not pickle-able, which raises a TypeError. Alternatively,
    one can make use of `deepcopy_component` to deepcopy a module containing used bijectors.
    """
    input = 1.
    _ = module(input)
    with pytest.raises(TypeError):
        deepcopy(module)
    module_copy = deepcopy_components(module)
    assert module.var == module_copy.var
    assert module.var is not module_copy.var
    module_copy.var.assign([5.0])
    assert module.var != module_copy.var
