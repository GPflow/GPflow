from pathlib import Path
from typing import Any, List, Optional, Sequence, Union

import numpy as np
import pytest
import tensorflow as tf
from _pytest.capture import CaptureFixture
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import gpflow
from gpflow.experimental.check_shapes import check_shape as cs
from gpflow.experimental.check_shapes import check_shapes
from gpflow.models import GPR, GPModel
from gpflow.monitor import (
    ExecuteCallback,
    ImageToTensorBoard,
    ModelToTensorBoard,
    Monitor,
    MonitorTask,
    MonitorTaskGroup,
    ScalarToTensorBoard,
)


class Data:
    num_data = 20
    num_steps = 2


class DummyTask(MonitorTask):
    def run(self, **kwargs: Any) -> None:
        pass


class DummyStepCallback:
    current_step = 0

    @check_shapes(
        "variables[all]: [...]",
        "values[all]: [...]",
    )
    def callback(
        self, step: int, variables: Sequence[tf.Variable], values: Sequence[tf.Tensor]
    ) -> None:
        self.current_step = step


@pytest.fixture
@check_shapes()
def model() -> GPModel:
    data = (
        cs(np.random.randn(Data.num_data, 2), "[N, 2]"),
        cs(np.random.randn(Data.num_data, 2), "[N, 2]"),
    )
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0])
    return GPR(data, kernel, noise_variance=0.01)


@pytest.fixture
def monitor(model: GPModel, tmp_path: Path) -> Monitor:
    tmp_path_str = str(tmp_path)

    @check_shapes(
        "return: []",
    )
    def lml_callback() -> tf.Tensor:
        return model.log_marginal_likelihood()

    def print_callback() -> None:
        print("foo")

    return Monitor(
        MonitorTaskGroup(
            [
                ModelToTensorBoard(tmp_path_str, model),
                ScalarToTensorBoard(tmp_path_str, lml_callback, "lml"),
            ],
            period=2,
        ),
        MonitorTaskGroup(ExecuteCallback(print_callback), period=1),
    )


def _get_size_directory(d: Path) -> int:
    """Calculating the size of a directory (in Bytes)."""
    return sum(f.stat().st_size for f in d.glob("**/*"))


# Smoke tests for the individual tasks
# #####################################


def test_ExecuteCallback() -> None:
    def callback() -> None:
        print("ExecuteCallback test")

    task = ExecuteCallback(callback)
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ImageToTensorBoard(tmp_path: Path) -> None:
    """Smoke test `ImageToTensorBoard` in Eager and Compiled mode."""
    tmp_path_str = str(tmp_path)

    def plotting_cb(fig: Figure, axes: Axes) -> None:
        axes[0, 0].plot(np.random.randn(2), np.random.randn(2))
        axes[1, 0].plot(np.random.randn(2), np.random.randn(2))
        axes[0, 1].plot(np.random.randn(2), np.random.randn(2))
        axes[1, 1].plot(np.random.randn(2), np.random.randn(2))

    fig_kwargs = dict(figsize=(10, 10))
    subplots_kwargs = dict(sharex=True, nrows=2, ncols=2)
    task = ImageToTensorBoard(
        tmp_path_str, plotting_cb, "image", fig_kw=fig_kwargs, subplots_kw=subplots_kwargs
    )

    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ScalarToTensorBoard(tmp_path: Path) -> None:
    """Smoke test `ScalarToTensorBoard` in Eager and Compiled mode."""
    tmp_path_str = str(tmp_path)

    def scalar_cb() -> float:
        return 0.0

    task = ScalarToTensorBoard(tmp_path_str, scalar_cb, "scalar")
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ScalarToTensorBoard_with_argument(tmp_path: Path) -> None:
    """Smoke test `ScalarToTensorBoard` in Eager and Compiled mode."""
    tmp_path_str = str(tmp_path)

    def scalar_cb(x: Optional[float] = None) -> float:
        assert x is not None
        return 2 * x

    task = ScalarToTensorBoard(tmp_path_str, scalar_cb, "scalar")
    compiled_task = tf.function(task)
    task(0, x=1.0)
    compiled_task(0, x=1.0)


def test_ScalarToTensorBoard_with_wrong_keyword_argument(tmp_path: Path) -> None:
    tmp_path_str = str(tmp_path)

    def scalar_cb(x: Optional[float] = None) -> float:
        assert x is not None
        return 2 * x

    task = ScalarToTensorBoard(tmp_path_str, scalar_cb, "scalar")
    compiled_task = tf.function(task)

    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'y'"):
        task(0, y=1.0)

    with pytest.raises(TypeError, match=r"got an unexpected keyword argument 'y'"):
        compiled_task(0, y=1.0)


def test_ModelToTensorboard(model: GPModel, tmp_path: Path) -> None:
    """Smoke test `ModelToTensorBoard` in Eager and Compiled mode."""
    tmp_path_str = str(tmp_path)
    task = ModelToTensorBoard(tmp_path_str, model)
    task(0)
    compiled_task = tf.function(task)
    compiled_task(0)


def test_ExecuteCallback_arguments(capsys: CaptureFixture[str]) -> None:
    def cb1(x: Optional[int] = None, **_: Any) -> None:
        assert x is not None
        print(x)

    def cb2(**_: Any) -> None:
        print(2)

    def cb3(y: Optional[int] = None, **_: Any) -> None:
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


def none() -> None:
    return None


@pytest.mark.parametrize(
    "task_or_tasks",
    [
        ExecuteCallback(none),
        [ExecuteCallback(none)],
        [ExecuteCallback(none), ExecuteCallback(none)],
    ],
)
def test_MonitorTaskGroup_and_Monitor(task_or_tasks: Union[MonitorTask, List[MonitorTask]]) -> None:
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


def test_Monitor(monitor: Monitor) -> None:
    monitor(0)
    compiled_monitor = tf.function(monitor)
    compiled_monitor(0)


# Functionality tests
# ###################


def test_compiled_execute_callable(capsys: CaptureFixture[str]) -> None:
    """
    Test that the `ExecuteCallback` when compiled behaves as expected.
    We test that python prints are not executed anymore.
    """
    string_to_print = "Eager mode"

    def callback() -> None:
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


def test_periodicity_group(capsys: CaptureFixture[str]) -> None:
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


def test_logdir_created(monitor: Monitor, model: GPModel, tmp_path: Path) -> None:
    """
    Check that TensorFlow summaries are written.
    """
    # check existence
    assert tmp_path.is_dir()
    size_before = _get_size_directory(tmp_path)
    assert size_before > 0

    opt = tf.optimizers.Adam()
    for step in range(Data.num_steps):
        opt.minimize(model.training_loss, model.trainable_variables)
        monitor(step)

    size_after = _get_size_directory(tmp_path)
    assert size_after > size_before


def test_compile_monitor(monitor: Monitor, model: GPModel) -> None:
    opt = tf.optimizers.Adam()

    @tf.function
    def tf_func(step: tf.Tensor) -> None:
        opt.minimize(model.training_loss, model.trainable_variables)
        monitor(step)

    for step in tf.range(100):
        tf_func(step)


def test_scipy_monitor(monitor: Monitor, model: GPModel) -> None:
    opt = gpflow.optimizers.Scipy()

    opt.minimize(model.training_loss, model.trainable_variables, step_callback=monitor)


def test_scipy_monitor_called(model: GPModel) -> None:
    task = DummyTask()
    monitor = Monitor(MonitorTaskGroup(task, period=1))
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, step_callback=monitor)
    assert task.current_step > 1


def test_scipy_step_callback_called(model: GPModel) -> None:
    dsc = DummyStepCallback()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(model.training_loss, model.trainable_variables, step_callback=dsc.callback)
    assert dsc.current_step > 1
