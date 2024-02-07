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
import copy
import unittest
import warnings
from typing import Any, Callable, Dict, List, Sequence, Tuple
from unittest.mock import MagicMock

import numpy as np
import pytest
import tensorflow as tf
from packaging.version import Version

import gpflow
from gpflow.base import AnyNDArray
from gpflow.config import default_float
from gpflow.models import GPR, GPModel
from gpflow.optimizers.scipy import LossClosure

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
                m.training_loss,
                m.trainable_variables,
                compile=compile,
                tf_fun_args=tf_fun_args,
            )
    else:
        opt.minimize(
            m.training_loss,
            m.trainable_variables,
            compile=compile,
            tf_fun_args=tf_fun_args,
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


def _loss_closure_builder(counter: List[int], inputs: Sequence[tf.Variable]) -> LossClosure:
    def closure() -> tf.Tensor:
        counter[0] += 1
        return tf.reduce_sum(tf.concat([tf.reshape(i, (-1,)) for i in inputs], axis=0) ** 2)

    return closure


def _create_variables() -> Sequence[tf.Variable]:
    return [
        tf.Variable(tf.range(5, dtype=default_float())),
        tf.Variable(tf.range(10, dtype=default_float())),
    ]


def _get_eval_func_closure(eval_func: Callable[[AnyNDArray], Tuple[AnyNDArray, AnyNDArray]]) -> Any:
    assert eval_func.__closure__ is not None  # type: ignore[attr-defined]
    return eval_func.__closure__[0].cell_contents  # type: ignore[attr-defined]


# This test checks the basic cache behaviour of the Scipy optimizer:
#   the cache is used when the same arguments are passed to the minimize method multiple times
#   without compile=True, the cache is not affected
def test_scipy__cache_behaviour() -> None:
    opt = gpflow.optimizers.Scipy()
    variables1 = _create_variables()
    variables2 = _create_variables()
    counter1 = [0]  # Each closure has its own counter
    counter2 = [0]
    closure1 = _loss_closure_builder(counter1, variables1)
    closure2 = _loss_closure_builder(counter2, variables2)

    # Ensure the cache is empty
    assert len(opt.compile_cache) == 0

    # Call eval_func multiple times with the same arguments
    for _ in range(4):
        opt.minimize(
            closure1, variables1, compile=True, tf_fun_args={}, allow_unused_variables=False
        )
        # Check that the cache was used
        assert len(opt.compile_cache) == 1

    # Get the key of the cached tf.function
    key = list(opt.compile_cache.keys())[0]

    # Assert that the key is as expected
    assert key == (
        closure1,
        tuple(id(var) for var in variables1),
        frozenset({}.items()),
        False,
    )

    # Call eval_func multiple times with the same arguments, but without compile. This should
    # not affect the cache.
    for _ in range(4):
        opt.minimize(
            closure1, variables1, compile=False, tf_fun_args={}, allow_unused_variables=False
        )
        # Check that the cache size did not change
        assert len(opt.compile_cache) == 1

    # Fill the cache with closure2
    for _ in range(4):
        opt.minimize(
            closure2, variables2, compile=True, tf_fun_args={}, allow_unused_variables=False
        )
        # Check that the cache size did change
        assert len(opt.compile_cache) == 2

    # The closure should be run
    #   1 (during the first compilation) + 4 (calls without compilation) times
    assert counter1[0] == 1 + 4
    # The closure should only be run once (during the first compilation)
    assert counter2[0] == 1


# This test fills the cache, and then adds one more entry to force the cache to replace the oldest
# entry.
@pytest.mark.parametrize("compile_cache_size", [1, 2, 3])
def test_scipy__cache_replacement(compile_cache_size: int) -> None:
    opt = gpflow.optimizers.Scipy(compile_cache_size=compile_cache_size)

    # Create more closures and variables than the cache can hold
    variables = [tf.Variable(float(i)) for i in range(compile_cache_size + 1)]
    closures = [_loss_closure_builder([0], var) for var in variables]

    # Fill the cache
    for i in range(compile_cache_size):
        opt.eval_func(closures[i], [variables[i]], tf_fun_args={})
        assert len(opt.compile_cache) == i + 1

    # Add one more closure to force the cache to replace the oldest entry
    opt.eval_func(closures[-1], [variables[-1]], tf_fun_args={})
    assert len(opt.compile_cache) == compile_cache_size
    assert set(opt.compile_cache.keys()) == set(
        [
            (closures[i], (id(variables[i]),), frozenset(), False)
            for i in range(1, compile_cache_size + 1)
        ]
    )


# This test checks that the cache is disabled when compile_cache_size is set to 0.
def test_scipy__cache_disabled() -> None:
    opt = gpflow.optimizers.Scipy(compile_cache_size=0)
    variables = _create_variables()
    counter = [0]
    closure = _loss_closure_builder(counter, variables)

    # Ensure the cache is empty
    assert len(opt.compile_cache) == 0

    # Call eval_func multiple times with the same arguments
    for _ in range(4):
        opt.minimize(closure, variables, compile=True, tf_fun_args={}, allow_unused_variables=False)
        # Check that the cache was not used
        assert len(opt.compile_cache) == 0

    # The closure should be run every time
    assert counter[0] == 4


# This test ensures an error is raised when compile_cache_size is set to a negative value.
def test_scipy__cache_raises_negative_size() -> None:
    with pytest.raises(ValueError, match=r"The 'compile_cache_size' argument must be non-negative"):
        gpflow.optimizers.Scipy(compile_cache_size=-1)


# This test ensures that the cache behaves correctly considering all arguments.
# It verifies that the cache has a miss when any argument changes, and a hit when all arguments
# remain the same. Note that when `compile` is `False`, the cache is not used and we expect a miss.
@pytest.mark.parametrize(
    "expect_cache_hit, same_closure2, same_variables2, same_tf_fun_args2, "
    "allow_unused_variables2, compile2",
    [
        pytest.param(
            True,
            True,
            True,
            True,
            False,
            True,
            id="hit: same closure & variables, same remaining args",
        ),
        pytest.param(
            False,
            True,
            False,
            True,
            False,
            True,
            id="miss: same closure, different variables, same remaining args",
        ),
        pytest.param(
            False,
            False,
            True,
            True,
            False,
            True,
            id="miss: different closure, same variables, same remaining args",
        ),
        pytest.param(
            False,
            True,
            True,
            False,
            False,
            True,
            id="miss: same closure & variables, different tf_fun_args",
        ),
        pytest.param(
            False,
            True,
            True,
            True,
            True,
            True,
            id="miss: same closure & variables, different allow_unused_variables",
        ),
        pytest.param(
            False,
            True,
            True,
            True,
            False,
            False,
            id="miss: same closure, variables & args, but no-compile",
        ),
    ],
)
def test_scipy__cache_hit_miss(
    expect_cache_hit: bool,
    same_closure2: bool,
    same_variables2: bool,
    same_tf_fun_args2: bool,
    allow_unused_variables2: bool,
    compile2: bool,
) -> None:
    opt = gpflow.optimizers.Scipy()

    variables1 = _create_variables()
    counter1 = [0]
    closure1 = _loss_closure_builder(counter1, variables1)
    tf_fun_args1 = dict(experimental_relax_shapes=True)
    dummy_x = tf.concat([tf.zeros_like(var) for var in variables1], 0)

    if same_closure2:
        closure2 = closure1
    else:
        closure2 = _loss_closure_builder([0], variables1)

    if same_variables2:
        variables2 = variables1
    else:
        variables2 = [variables1[0]]  # Pick a subset of variables

    if same_tf_fun_args2:
        tf_fun_args2 = dict(experimental_relax_shapes=True)
    else:
        tf_fun_args2 = dict(experimental_relax_shapes=False)

    # Populate cache with first closure and variables
    eval_func1 = opt.eval_func(closure1, variables1, tf_fun_args1, True, False)
    eval_func1(dummy_x)

    # Call with passed in arguments
    eval_func2 = opt.eval_func(
        closure2, variables2, tf_fun_args2, compile2, allow_unused_variables2
    )
    eval_func2(dummy_x)

    if expect_cache_hit:
        # Check that the cache was used
        assert len(opt.compile_cache) == 1
        assert _get_eval_func_closure(eval_func2) is _get_eval_func_closure(
            eval_func1
        ), "Cache hit expected"
    else:
        # Check that the cache was bypassed and a new compiled function was created, if compile=True
        if compile2:
            assert len(opt.compile_cache) == 2
        else:
            assert len(opt.compile_cache) == 1
        assert _get_eval_func_closure(eval_func2) is not _get_eval_func_closure(
            eval_func1
        ), "Cache miss expected"

    # The original closure, variables and args should still be in the cache
    eval_func3 = opt.eval_func(closure1, variables1, tf_fun_args1, True, False)
    eval_func3(dummy_x)
    assert _get_eval_func_closure(eval_func3) is _get_eval_func_closure(
        eval_func1
    ), "Cache hit expected"

    # The closures should be run
    #   1 (during the first compilation) + 1 (if miss and closure was reused) times
    exp_count = 1 + (1 if not expect_cache_hit and same_closure2 else 0)
    assert counter1[0] == exp_count


# In the first test, we minimize the same model twice and check that this reuses the existing
# compiled function in the cache.
def test_scipy__cache_with_same_model() -> None:
    model = _create_full_gp_model()
    opt = gpflow.optimizers.Scipy()

    loss_closure = model.training_loss

    # Call minimize twice with the same model
    opt.minimize(
        loss_closure,
        model.trainable_variables,
        compile=True,
        tf_fun_args={jit_compile_arg: True},
    )
    opt.minimize(
        loss_closure,
        model.trainable_variables,
        compile=True,
        tf_fun_args={jit_compile_arg: True},
    )

    # Check that the cache was used
    assert len(opt.compile_cache) == 1


# In the second test, we minimize two slightly different models (one has one fewer trainable
# variable), and check that this bypasses the cache and creates a new compiled function.
def test_scipy__cache_with_different_models() -> None:
    model = _create_full_gp_model()
    opt = gpflow.optimizers.Scipy()

    loss_closure = model.training_loss

    # Call minimize twice with slightly different models
    opt.minimize(
        loss_closure,
        model.trainable_variables,
        compile=True,
        tf_fun_args={jit_compile_arg: True},
    )
    # Set the lengthscales to be non-trainable
    gpflow.utilities.set_trainable(model.kernel.lengthscales, False)
    opt.minimize(
        loss_closure,
        model.trainable_variables,
        compile=True,
        tf_fun_args={jit_compile_arg: True},
    )

    # Check that the cache was bypassed and a new compiled function was created
    assert len(opt.compile_cache) == 2


def test_scipy__deep_copyable() -> None:
    opt = gpflow.optimizers.Scipy()
    variables1 = _create_variables()
    counter1 = [0]
    closure1 = _loss_closure_builder(counter1, variables1)

    # Ensure the cache is empty
    assert len(opt.compile_cache) == 0

    # Call eval_func multiple times with the same arguments
    for _ in range(4):
        opt.minimize(
            closure1, variables1, compile=True, tf_fun_args={}, allow_unused_variables=False
        )
        # Check that the cache was used
        assert len(opt.compile_cache) == 1

    # Check that the optimizer can still be deepcopied
    opt_copy = copy.deepcopy(opt)

    # The cache is not copied
    assert len(opt_copy.compile_cache) == 0

    # Call eval_func multiple times with the same arguments
    for _ in range(4):
        opt_copy.minimize(
            closure1, variables1, compile=True, tf_fun_args={}, allow_unused_variables=False
        )
        # Check that the cache was used
        assert len(opt_copy.compile_cache) == 1
