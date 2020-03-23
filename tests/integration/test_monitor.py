import os

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.monitor import (
    ModelToTensorBoard,
    ScalarToTensorBoard,
    ImageToTensorBoard,
    MonitorCollection,
)


class Data:
    num_data = 20
    num_steps = 10


@pytest.fixture
def model():
    data = (
        np.random.randn(Data.num_data, 2),  # [N, 2]
        np.random.randn(Data.num_data, 2),  # [N, 1]
    )
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0])
    model = gpflow.models.GPR(data, kernel, noise_variance=0.01)

    return model


def _get_size_directory(dir):
    """ Calculating the size of a directory (in Bytes) """
    return sum(d.stat().st_size for d in os.scandir(dir) if d.is_file())


def test_logdir_created(model, tmp_path):
    tmp_path = str(tmp_path)

    def elbo_callback():
        return model.log_likelihood()

    monitor = MonitorCollection(
        [ModelToTensorBoard(tmp_path, model), ScalarToTensorBoard(tmp_path, elbo_callback, "elbo"),]
    )

    # check existence
    assert os.path.exists(tmp_path) and os.path.isdir(tmp_path)
    size_before = _get_size_directory(tmp_path)
    assert size_before > 0

    @tf.function
    def closure():
        return -1.0 * model.log_likelihood()

    opt = tf.optimizers.Adam()
    for step in range(Data.num_steps):
        opt.minimize(closure, model.trainable_variables)
        monitor(step)

    size_after = _get_size_directory(tmp_path)
    assert size_after > size_before


def test_compile_monitor(model, tmp_path):
    tmp_path = str(tmp_path)
    monitor = MonitorCollection([ModelToTensorBoard(tmp_path, model)])

    opt = tf.optimizers.Adam()

    @tf.function
    def tf_func(step):
        closure = lambda: -1.0 * model.log_likelihood()
        opt.minimize(closure, model.trainable_variables)
        monitor(step)

    for step in tf.range(100):
        tf_func(step)


def test_ImageToTensorBoard(tmp_path):
    """ Smoke test `ImageToTensorBoard` in Eager and Compiled mode """
    tmp_path = str(tmp_path)

    def plotting_cb(fig, axes):
        axes[0, 0].plot(np.random.randn(2), np.random.randn(2), ".")
        axes[1, 0].plot(np.random.randn(2), np.random.randn(2), ".")
        axes[0, 1].plot(np.random.randn(2), np.random.randn(2), ".")
        axes[1, 1].plot(np.random.randn(2), np.random.randn(2), ".")

    fig_kwargs = dict(figsize=(10, 10))
    subplots_kwargs = dict(sharex=True, nrows=2, ncols=2)
    task = ImageToTensorBoard(
        tmp_path, plotting_cb, "image", fig_kw=fig_kwargs, subplots_kw=subplots_kwargs
    )

    task(0)


def test_ScalarToTensorBoard(tmp_path, capfd):
    """ Smoke test `ScalarToTensorBoard` in Eager and Compiled mode """
    tmp_path = str(tmp_path)

    def scalar_cb():
        return 0.0

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    compiled_task = tf.function(task.__call__)

    task(0)
    compiled_task(0)


def test_ScalarToTensorBoard_with_argument(tmp_path):
    """ Smoke test `ScalarToTensorBoard` in Eager and Compiled mode """
    tmp_path = str(tmp_path)

    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    compiled_task = tf.function(task.__call__)
    task(0, x=1.0)
    compiled_task(0, x=1.0)

    tasks = MonitorCollection([task])
    compiled_tasks = tf.function(tasks.__call__)
    tasks(0, x=1.0)
    compiled_tasks(0, x=1.0)


def test_ScalarToTensorBoard_with_wrong_kw_argument(tmp_path):
    tmp_path = str(tmp_path)

    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoard(tmp_path, scalar_cb, "scalar")
    tasks = MonitorCollection([task])
    compiled_tasks = tf.function(tasks.__call__)

    with pytest.raises(TypeError, match=r".*got an unexpected keyword argument 'y'.*"):
        tasks(0, y=1.0)

    with pytest.raises(TypeError, match=r".*got an unexpected keyword argument 'y'.*"):
        compiled_tasks(0, y=1.0)


def test_ModelToTensorboard(model, tmp_path):
    """ Smoke test `ModelToTensorBoard` in Eager and Compiled mode """
    tmp_path = str(tmp_path)
    task = ModelToTensorBoard(tmp_path, model)
    compiled_task = tf.function(task.__call__)
    task(0)
    compiled_task(0)
