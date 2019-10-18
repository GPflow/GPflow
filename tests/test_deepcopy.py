from copy import deepcopy
import pytest

import tensorflow as tf
import tensorflow_probability as tfp
from gpflow.utilities import deepcopy_components



class A(tf.Module):
    def __init__(self):
        self.var = tf.Variable(tf.random.normal((1,)))
        self.bijector = tfp.bijectors.Softplus()

    def __call__(self, x):
        return self.bijector(x)


class B(tf.Module):
    def __init__(self):
        self.var = tf.Variable(tf.random.normal((1,)))
        self.a = A()

    def __call__(self, x):
        return self.a(x)


@pytest.mark.parametrize('module', [A(), B()])
def test_deepcopy_component_works(module):
    input = 1.
    _ = module(input)
    with pytest.raises(TypeError):
        deepcopy(module)
    module_copy = deepcopy_components(module)
    assert module.var == module_copy.var