# Copyright 2019 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.base import AnyNDArray
from gpflow.config import default_float
from gpflow.models import GPR, GPModel

rng = np.random.RandomState(0)


class Datum:
    X: AnyNDArray = rng.rand(20, 1) * 10
    Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
    Y = np.tile(Y, 2)  # two identical columns
    Xtest: AnyNDArray = rng.rand(10, 1) * 10
    data = (X, Y)


def _create_full_gp_model() -> GPModel:
    """
    GP Regression
    """
    return GPR(
        (Datum.X, Datum.Y),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=gpflow.mean_functions.Constant(),
    )


def test_scipy_jit() -> None:
    m1 = _create_full_gp_model()
    m2 = _create_full_gp_model()

    opt1 = gpflow.optimizers.Scipy()
    opt2 = gpflow.optimizers.Scipy()

    opt1.minimize(
        m1.training_loss,
        variables=m1.trainable_variables,
        options=dict(maxiter=50),
        compile=False,
    )
    opt2.minimize(
        m2.training_loss,
        variables=m2.trainable_variables,
        options=dict(maxiter=50),
        compile=True,
    )

    def get_values(model: GPModel) -> AnyNDArray:
        return np.array([var.numpy().squeeze() for var in model.trainable_variables])

    np.testing.assert_allclose(get_values(m1), get_values(m2), rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("compile", [True, False])
def test_scipy__optimal(compile: bool) -> None:
    target1 = [0.2, 0.8]
    target2 = [0.6]
    v1 = tf.Variable([0.5, 0.5], dtype=default_float())
    v2 = tf.Variable([0.5], dtype=default_float())
    compilation_count = 0

    def f() -> tf.Tensor:
        nonlocal compilation_count
        compilation_count += 1
        return tf.reduce_sum((target1 - v1) ** 2) + tf.reduce_sum((target2 - v2) ** 2)

    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(f, [v1, v2], compile=compile)

    if compile:
        assert 1 == compilation_count
    else:
        assert 1 < compilation_count
    assert result.success
    np.testing.assert_allclose(target1 + target2, result.x)
    np.testing.assert_allclose(target1, v1)
    np.testing.assert_allclose(target2, v2)


@pytest.mark.parametrize("compile", [True, False])
def test_scipy__partially_disconnected_variable(compile: bool) -> None:
    target1 = 0.2
    target2 = 0.6
    v1 = tf.Variable([0.5, 0.5], dtype=default_float())
    v2 = tf.Variable(0.5, dtype=default_float())

    def f() -> tf.Tensor:
        # v1[1] not used.
        v10 = v1[0]
        return (target1 - v10) ** 2 + (target2 - v2) ** 2

    opt = gpflow.optimizers.Scipy()
    result = opt.minimize(f, [v1, v2], compile=compile)

    assert result.success
    np.testing.assert_allclose([target1, 0.5, target2], result.x)
    np.testing.assert_allclose([target1, 0.5], v1)
    np.testing.assert_allclose(target2, v2)


@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize("allow_unused_variables", [True, False])
def test_scipy__disconnected_variable(compile: bool, allow_unused_variables: bool) -> None:
    target1 = [0.2, 0.8]
    v1 = tf.Variable([0.5, 0.5], dtype=default_float(), name="v1")
    v2 = tf.Variable([0.5], dtype=default_float(), name="v2")

    def f() -> tf.Tensor:
        # v2 not used.
        return tf.reduce_sum((target1 - v1) ** 2)

    opt = gpflow.optimizers.Scipy()

    if allow_unused_variables:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = opt.minimize(
                f, [v1, v2], compile=compile, allow_unused_variables=allow_unused_variables
            )

        (warning,) = w
        message = warning.message
        assert isinstance(message, Warning)
        msg = message.args[0]
        assert v2.name in msg

        assert result.success
        np.testing.assert_allclose(target1 + [0.5], result.x)
        np.testing.assert_allclose(target1, v1)
        np.testing.assert_allclose([0.5], v2)
    else:
        with pytest.raises(ValueError, match=v2.name):
            opt.minimize(f, [v1, v2], allow_unused_variables=allow_unused_variables)
