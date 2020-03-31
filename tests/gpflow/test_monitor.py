import os
from typing import List

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.monitor import (
    ExecuteCallback,
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)


class Data:
    num_data = 20
    num_steps = 2


@pytest.fixture
def model():
    data = (
        np.random.randn(Data.num_data, 2),  # [N, 2]
        np.random.randn(Data.num_data, 2),  # [N, 1]
    )
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0])
    return gpflow.models.GPR(data, kernel, noise_variance=0.01)


@pytest.fixture
def monitor(model, tmp_path):
    tmp_path = str(tmp_path)

    def lml_callback():
        return model.log_marginal_likelihood()

    def print_callback():
        print("foo")

    return Monitor(
        MonitorTaskGroup(
            [
                ModelToTensorBoard(tmp_path, model),
                ScalarToTensorBoard(tmp_path, lml_callback, "lml"),
            ],
            period=2,
        ),
        MonitorTaskGroup(ExecuteCallback(print_callback), period=1),
    )


def _get_size_directory(dir):
    """Calculating the size of a directory (in Bytes)."""
    return sum(d.stat().st_size for d in os.scandir(dir) if d.is_file())


# Smoke tests for the individual tasks
# #####################################


def test_ExecuteCallback():
    def callback():
        print("ExecuteCallback test")

    task = ExecuteCallback(callback)
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ImageToTensorBoard(tmp_path):
    """Smoke test `ImageToTensorBoard` in Eager and Compiled mode."""
    tmp_path = str(tmp_path)

    def plotting_cb(fig, axes):
        axes[0, 0].plot(np.random.randn(2), np.random.randn(2))
        axes[1, 0].plot(np.random.randn(2), np.random.randn(2))
        axes[0, 1].plot(np.random.randn(2), np.random.randn(2))
        axes[1, 1].plot(np.random.randn(2), np.random.randn(2))

    fig_kwargs = dict(figsize=(10, 10))
    subplots_kwargs = dict(sharex=True, nrows=2, ncols=2)
    task = ImageToTensorBoard(
        tmp_path, plotting_cb, "image", fig_kw=fig_kwargs, subplots_kw=subplots_kwargs
    )

    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ScalarToTensorBoard(tmp_path):
    """Smoke test `ScalarToTensorBoard` in Eager and Compiled mode."""
    tmp_path = str(tmp_path)

    def scalar_cb():
        return 0.0

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ScalarToTensorBoard_with_argument(tmp_path):
    """Smoke test `ScalarToTensorBoard` in Eager and Compiled mode."""
    tmp_path = str(tmp_path)

    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    compiled_task = tf.function(task)
    task(0, x=1.0)
    compiled_task(0, x=1.0)


def test_ScalarToTensorBoard_with_wrong_keyword_argument(tmp_path):
    tmp_path = str(tmp_path)

    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    compiled_task = tf.function(task)

    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'y'"):
        task(0, y=1.0)

    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'y'"):
        compiled_task(0, y=1.0)


def test_ModelToTensorboard(model, tmp_path):
    """Smoke test `ModelToTensorBoard` in Eager and Compiled mode."""
    tmp_path = str(tmp_path)
    task = ModelToTensorBoard(tmp_path, model)
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ExecuteCallback_arguments(capsys):
    def cb1(x=None, **_):
        assert x is not None
        print(x)

    def cb2(**_):
        print(2)

    def cb3(y=None, **_):
        assert y is not None
        print(y)

    group1 = MonitorTaskGroup([ExecuteCallback(cb1), ExecuteCallback(cb2)])
    group2 = MonitorTaskGroup(ExecuteCallback(cb3))
    monitor = Monitor(group1, group2)
    monitor(0, x=1, y=3)
    out, _ = capsys.readouterr()
    assert out == "1\n2\n3\n"


# Smoke test Monitor and MonitorTaskGroup
# ########################################


@pytest.mark.parametrize(
    "task_or_tasks",
    [
        ExecuteCallback(lambda: 0.0),
        [ExecuteCallback(lambda: 0.0)],
        [ExecuteCallback(lambda: 0.0), ExecuteCallback(lambda: 0.0)],
    ],
)
def test_MonitorTaskGroup_and_Monitor(task_or_tasks):
    group = MonitorTaskGroup(task_or_tasks, period=2)

    # check that the tasks is actually a list (custom setter)
    isinstance(group.tasks, list)

    # Smoke test the __call__
    group(0)
    compiled_group = tf.function(group)
    compiled_group(0)

    # Smoke test the Monitor wrapper
    monitor = Monitor(group)
    monitor(0)
    compiled_monitor = tf.function(monitor)
    compiled_monitor(0)


def test_Monitor(monitor):
    monitor(0)
    compiled_monitor = tf.function(monitor)
    compiled_monitor(0)


# Functionality tests
# ###################


def test_compiled_execute_callable(capsys):
    """
    Test that the `ExecuteCallback` when compiled behaves as expected.
    We test that python prints are not executed anymore.
    """
    string_to_print = "Eager mode"

    def callback():
        print(string_to_print)

    task = ExecuteCallback(callback)

    # Eager mode
    for i in range(Data.num_steps):
        task(i)
    out, _ = capsys.readouterr()

    # We expect a print for each step
    assert out == (f"{string_to_print}\n" * Data.num_steps)

    # Autograph mode
    compiled_task = tf.function(task)
    for i in tf.range(Data.num_steps):
        compiled_task(i)
    out, _ = capsys.readouterr()
    assert out == f"{string_to_print}\n"


def test_periodicity_group(capsys):
    """Test that groups are called at different periods."""

    task_a = ExecuteCallback(lambda: print("a", end=" "))
    task_b = ExecuteCallback(lambda: print("b", end=" "))
    task_X = ExecuteCallback(lambda: print("X", end=" "))

    group_often = MonitorTaskGroup([task_a, task_b], period=1)
    group_seldom = MonitorTaskGroup([task_X], period=3)
    monitor = Monitor(group_often, group_seldom)
    for i in range(7):
        monitor(i)

    out, _ = capsys.readouterr()
    expected = "a b X a b a b a b X a b a b a b X "
    assert out == expected

    # AutoGraph mode
    compiled_monitor = tf.function(monitor)
    for i in tf.range(7):
        compiled_monitor(i)

    # When using TF's range and compiling the monitoring we only expected the python prints once.
    out, _ = capsys.readouterr()
    assert "a b X"


def test_logdir_created(monitor, model, tmp_path):
    """
    Check that TensorFlow summaries are written.
    """
    tmp_path = str(tmp_path)

    # check existence
    assert os.path.exists(tmp_path) and os.path.isdir(tmp_path)
    size_before = _get_size_directory(tmp_path)
    assert size_before > 0

    opt = tf.optimizers.Adam()
    for step in range(Data.num_steps):
        opt.minimize(model.training_loss, model.trainable_variables)
        monitor(step)

    size_after = _get_size_directory(tmp_path)
    assert size_after > size_before


def test_compile_monitor(monitor, model):
    opt = tf.optimizers.Adam()

    @tf.function
    def tf_func(step):
        opt.minimize(model.training_loss, model.trainable_variables)
        monitor(step)

    for step in tf.range(100):
        tf_func(step)
