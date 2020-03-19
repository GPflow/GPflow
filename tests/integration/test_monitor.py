import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.utilities.monitor import (
    ModelToTensorBoardTask,
    ScalarToTensorBoardTask,
    TasksCollection,
    ImageToTensorBoardTask,
)


class Data:
    num_data = 20
    num_steps = 10
    log_dir = tempfile.gettempdir()


@pytest.fixture
def model_and_closure():
    data = (
        np.random.randn(Data.num_data, 2),  # [N, 2]
        np.random.randn(Data.num_data, 2),  # [N, 1]
    )
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[1.0, 2.0])
    model = gpflow.models.GPR(data, kernel, noise_variance=0.01)

    @tf.function
    def closure():
        return -1.0 * model.log_likelihood()

    return model, closure


def _get_size_directory(dir):
    """ Calculating the size of a directory (in Bytes) """
    return sum(d.stat().st_size for d in os.scandir(dir) if d.is_file())


def test_logdir_created(model_and_closure):
    model, closure = model_and_closure

    def elbo_callback():
        return model.log_likelihood()

    tasks = TasksCollection(
        [
            ModelToTensorBoardTask(Data.log_dir, model),
            ScalarToTensorBoardTask(Data.log_dir, elbo_callback, "elbo"),
        ]
    )

    # check existence
    assert os.path.exists(Data.log_dir) and os.path.isdir(Data.log_dir)
    size_before = _get_size_directory(Data.log_dir)
    assert size_before > 0

    opt = tf.optimizers.Adam()
    for step in range(Data.num_steps):
        opt.minimize(closure, model.trainable_variables)
        tasks(step)

    size_after = _get_size_directory(Data.log_dir)
    assert size_after > size_before


def test_ImageToTensorBoardTask():
    def plotting_cb(fig, axes):
        axes[0, 0].plot(np.random.randn(100), np.random.randn(100), ".")
        axes[1, 0].plot(np.random.randn(100), np.random.randn(100), ".")
        axes[0, 1].plot(np.random.randn(100), np.random.randn(100), ".")
        axes[1, 1].plot(np.random.randn(100), np.random.randn(100), ".")

    fig_kwargs = dict(figsize=(10, 10))
    subplots_kwargs = dict(sharex=True, nrows=2, ncols=2)
    task = ImageToTensorBoardTask(
        Data.log_dir, plotting_cb, "image", fig_kw=fig_kwargs, subplots_kw=subplots_kwargs
    )

    task(0)


def test_ScalarToTensorBoardTask():
    def scalar_cb():
        return 0.0

    task = ScalarToTensorBoardTask(Data.log_dir, scalar_cb, "scalar")
    task(0)


def test_ScalarToTensorBoardTask_with_argument():
    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoardTask(Data.log_dir, scalar_cb, "scalar")
    task(0, x=1.0)

    tasks = TasksCollection([task])
    tasks(0, x=1.0)


def test_ScalarToTensorBoardTask_with_wrong_kw_argument():
    def scalar_cb(x=None):
        return 2 * x

    task = ScalarToTensorBoardTask(Data.log_dir, scalar_cb, "scalar")
    tasks = TasksCollection([task])

    with pytest.raises(TypeError, match=r".*got an unexpected keyword argument 'y'.*"):
        tasks(0, y=1.0)


def test_ModelToTensboardTask(model_and_closure):
    model, _ = model_and_closure
    task = ModelToTensorBoardTask(Data.log_dir, model)
    task(0)
