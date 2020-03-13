import copy

import gpflow
import pytest
import tensorflow as tf
import tensorflow_probability as tfp


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


class NestedModule(tf.Module):
    def __init__(self, module: tf.Module):
        self.module = module


@pytest.mark.parametrize("module", [A(), B()])
def test_clears_bijector_cache_and_deepcopy(module):
    """
    With each forward pass through a bijector, a cache is stored inside which prohibits the deepcopy of the bijector.
    This is due to the fact that HashableWeakRef objects are not pickle-able, which raises a TypeError. Alternatively,
    one can make use of `deepcopy_component` to deepcopy a module containing used bijectors.
    """
    input = 1.0
    _ = module(input)
    with pytest.raises(TypeError):
        copy.deepcopy(module)
    module_copy = gpflow.utilities.deepcopy(module)
    assert module.var == module_copy.var
    assert module.var is not module_copy.var
    module_copy.var.assign([5.0])
    assert module.var != module_copy.var


def test_freeze():
    module = NestedModule(NestedModule(A()))
    module_frozen = gpflow.utilities.freeze(module)
    assert len(module.variables) == 2
    assert module_frozen.variables == ()
    assert isinstance(module.module.module.var, tf.Variable)
    assert isinstance(module_frozen.module.module.var, tf.Tensor)


@pytest.mark.parametrize("path", ["kernel[wrong_index]", "kernel.non_existing_attr", "kernel[100]"])
def test_failures_getset_by_path(path):
    d = (tf.random.normal((10, 1)), tf.random.normal((10, 1)))
    k = gpflow.kernels.RBF() * gpflow.kernels.RBF()
    m = gpflow.models.GPR(d, kernel=k)

    with pytest.raises((ValueError, TypeError)):
        gpflow.utilities.getattr_by_path(m, path)

    if all([c in path for c in "[]"]):
        with pytest.raises((ValueError, TypeError)):
            gpflow.utilities.setattr_by_path(m, path, None)


@pytest.mark.parametrize(
    "path", ["kernel.kernels[0]", "kernel.kernels[0].variance", "kernel.kernels[0].lengthscale"]
)
def test_getset_by_path(path):
    d = (
        tf.random.normal((10, 1), dtype=gpflow.default_float()),
        tf.random.normal((10, 1), dtype=gpflow.default_float()),
    )
    k = gpflow.kernels.RBF() * gpflow.kernels.RBF()
    m = gpflow.models.GPR(d, kernel=k)

    gpflow.utilities.getattr_by_path(m, path)
    gpflow.utilities.setattr_by_path(m, path, None)
