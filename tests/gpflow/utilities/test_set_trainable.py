import tensorflow as tf

from gpflow import Parameter, set_trainable


def _module() -> tf.Module:
    class _Mod(tf.Module):
        def __init__(self) -> None:
            super().__init__()
            self.var = tf.Variable(0.0)
            self.param = Parameter(0.0)

    module = _Mod()

    assert len(module.trainable_variables) == 2
    assert len(module.variables) == 2

    return module


def test_can_set_not_trainable() -> None:
    module = _module()
    set_trainable(module, False)
    assert len(module.trainable_variables) == 0


def test_can_set_not_trainable_then_trainable_again() -> None:
    module = _module()
    set_trainable(module, False)
    set_trainable(module, True)
    assert len(module.trainable_variables) == len(module.variables)


def test_can_set_not_trainable_iterable() -> None:
    modules = [_module(), _module(), _module()]
    set_trainable(modules, False)
    assert all(len(m.trainable_variables) == 0 for m in modules)


def test_can_set_not_trainable_then_trainable_iterable() -> None:
    modules = [_module(), _module(), _module()]
    set_trainable(modules, False)
    set_trainable(modules, True)
    assert all(len(m.trainable_variables) == len(m.variables) for m in modules)
