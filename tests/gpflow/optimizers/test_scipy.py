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

import unittest
import warnings
from typing import Any, Dict
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

import gpflow
from gpflow.base import AnyNDArray
from gpflow.config import default_float
from gpflow.models import GPR, GPModel

rng = np.random.RandomState(0)

if Version(tf.__version__) >= Version("2.5"):
    jit_compile_arg = "jit_compile"
else:
    jit_compile_arg = "experimental_compile"


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
    m3 = _create_full_gp_model()

    opt1 = gpflow.optimizers.Scipy()
    opt2 = gpflow.optimizers.Scipy()
    opt3 = gpflow.optimizers.Scipy()

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
    opt3.minimize(
        m3.training_loss,
        variables=m3.trainable_variables,
        options=dict(maxiter=50),
        compile=True,
        tf_fun_args={jit_compile_arg: True},
    )

    def get_values(model: GPModel) -> AnyNDArray:
        return np.array([var.numpy().squeeze() for var in model.trainable_variables])

    np.testing.assert_allclose(get_values(m1), get_values(m2), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(get_values(m1), get_values(m3), rtol=1e-13, atol=1e-13)


@unittest.mock.patch("tensorflow.function")
@pytest.mark.parametrize("compile", [True, False])
@pytest.mark.parametrize(
    "tf_fun_args",
    [{}, dict(jit_compile=True), dict(jit_compile=False, other_arg="dummy")],
)
def test_scipy__tf_fun_args(
    mocked_tf_fun: MagicMock, compile: bool, tf_fun_args: Dict[str, Any]
) -> None:
    mocked_tf_fun.side_effect = lambda f, **_: f

    m = _create_full_gp_model()
    opt = gpflow.optimizers.Scipy()
    expect_raise = not compile and len(tf_fun_args)
    if expect_raise:
        with pytest.raises(
            ValueError, match="`tf_fun_args` should only be set when `compile` is True"
        ):
            opt.minimize(
                m.training_loss, m.trainable_variables, compile=compile, tf_fun_args=tf_fun_args
            )
    else:
        opt.minimize(
            m.training_loss, m.trainable_variables, compile=compile, tf_fun_args=tf_fun_args
        )

    if compile:
        received_args = mocked_tf_fun.call_args[1]
        expected_args = tf_fun_args
    else:
        # When no-compile, don't expect tf.function to be called.
        received_args = mocked_tf_fun.call_args
        expected_args = None
    assert received_args == expected_args


@pytest.mark.parametrize("compile,jit", [(True, True), (True, False), (False, False)])
def test_scipy__optimal(compile: bool, jit: bool) -> None:
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
    tf_fun_args = {jit_compile_arg: True} if jit else {}
    result = opt.minimize(f, [v1, v2], compile=compile, tf_fun_args=tf_fun_args)

    if compile:
        assert 1 == compilation_count
    else:
        assert 1 < compilation_count
    assert result.success
    np.testing.assert_allclose(target1 + target2, result.x)
    np.testing.assert_allclose(target1, v1)
    np.testing.assert_allclose(target2, v2)


@pytest.mark.parametrize("compile,jit", [(True, True), (True, False), (False, False)])
def test_scipy__partially_disconnected_variable(compile: bool, jit: bool) -> None:
    target1 = 0.2
    target2 = 0.6
    v1 = tf.Variable([0.5, 0.5], dtype=default_float())
    v2 = tf.Variable(0.5, dtype=default_float())

    def f() -> tf.Tensor:
        # v1[1] not used.
        v10 = v1[0]
        return (target1 - v10) ** 2 + (target2 - v2) ** 2

    opt = gpflow.optimizers.Scipy()
    tf_fun_args = {jit_compile_arg: True} if jit else {}
    result = opt.minimize(f, [v1, v2], compile=compile, tf_fun_args=tf_fun_args)

    assert result.success
    np.testing.assert_allclose([target1, 0.5, target2], result.x)
    np.testing.assert_allclose([target1, 0.5], v1)
    np.testing.assert_allclose(target2, v2)


@pytest.mark.parametrize("compile,jit", [(True, True), (True, False), (False, False)])
@pytest.mark.parametrize("allow_unused_variables", [True, False])
def test_scipy__disconnected_variable(
    compile: bool, jit: bool, allow_unused_variables: bool
) -> None:
    target1 = [0.2, 0.8]
    v1 = tf.Variable([0.5, 0.5], dtype=default_float(), name="v1")
    v2 = tf.Variable([0.5], dtype=default_float(), name="v2")

    def f() -> tf.Tensor:
        # v2 not used.
        return tf.reduce_sum((target1 - v1) ** 2)

    opt = gpflow.optimizers.Scipy()
    tf_fun_args = {jit_compile_arg: True} if jit else {}

    if allow_unused_variables:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = opt.minimize(
                f,
                [v1, v2],
                compile=compile,
                allow_unused_variables=allow_unused_variables,
                tf_fun_args=tf_fun_args,
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
            opt.minimize(
                f,
                [v1, v2],
                compile=compile,
                allow_unused_variables=allow_unused_variables,
                tf_fun_args=tf_fun_args,
            )
