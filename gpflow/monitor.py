# Copyright 2020 GPflow authors
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
""" Provides basic functionality to monitor optimisation runs """


from abc import ABC, abstractmethod
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Axes, Figure

from .models import BayesianModel
from .utilities import parameter_dict


class MonitorTask(ABC):
    """
    A base class for a monitoring task.

    All monitoring tasks are callable objects.
    A descendant class must implement the `run` method, which is the body of the monitoring task.
    """

    def __init__(self, period: int = 1):
        """
        :param period: Interval between triggering the task.
        """
        self.current_step = 0
        self.period = period

    def __call__(self, step: int, **kwargs):
        """
        It calls the 'run' function and sets the current step.

        :param step: current step in the optimisation.
        :param kwargs: additional key-word arguments that can be passed
            to the `run` method of the task. This is in particular handy for
            passing keyword argument to the callback of `ScalarToTensorBoard`.
        """
        self.current_step = tf.cast(step, tf.int64)
        if step % self.period == 0:
            self.run(**kwargs)

    @abstractmethod
    def run(self, **kwargs):
        """
        Implements the task to be executed on __call__.
        The current step is available through `self.current_step`.

        :param kwargs: keyword arguments available to the run method.
        """
        raise NotImplementedError


class ToTensorBoard(MonitorTask):
    def __init__(self, log_dir: str, period: int = 1):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested. E.g. ./logs/my_run/
        :param period: interval at which to run the task.
            For large values of`period` the task will be less frequently ran.
        """
        super().__init__(period)
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def __call__(self, step, **kwargs):
        with self.file_writer.as_default():
            super().__call__(step, **kwargs)
        self.file_writer.flush()


class ModelToTensorBoard(ToTensorBoard):
    """
    Monitoring task that creates a sensible TensorBoard for a model.

    By default, it writes to the TensorBoard all model parameters that are scalars,
    for parameter arrays (e.g. kernel.lengthscales) the values are sent to the TensorBoard
    if the array is smaller than 3. This behaviour can be adjusted using `max_size`.
    """

    def __init__(self, log_dir: str, model: BayesianModel, max_size: int = 3, period: int = 1):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param model: model to be monitord.
        :param max_size: maximum size of arrays (incl.) to store each
            element of the array independently as a scalar in the TensorBoard.
        :param period: interval at which to run the task.
            For large values of`period` the task will be less frequently ran.
        """
        super().__init__(log_dir, period)
        self.model = model
        self.max_size = max_size

    def run(self, **unused_kwargs):
        for k, v in parameter_dict(self.model).items():
            name = k.lstrip(".")  # keys are prepended with a '.', which we strip
            self._summarize_parameter(name, v)

    def _summarize_parameter(self, name: str, value: Union[float, np.ndarray]):
        """
        :param name: identifier used in tensorboard
        :param value: value to be stored in tensorboard
        """
        # tf.summary.scalar("hello", self.model.likelihood.variance, step=self.current_step)
        # tf.print("hello")
        # if isinstance(value, float):
        # print(tf.size(value))
        # tf.print(tf.size(value))
        value = tf.reshape(value, (-1,))
        if tf.size(value) > self.max_size:
            return

        i = 0
        for v in value:
            self._summarize_scalar_parameter(f"{name}[{i}]", v)
            i += 1
        # if tf.size(value) == 1:
        #     print(value)
        #     tf.print(value)
        #     self._summarize_scalar_parameter(name, value)
        # if tf.size(value) <= self.max_size:
        #     # elif isinstance(value, np.ndarray) and value.size <= self.max_size:
        #     self._summarize_array_parameter(name, value)

    def _summarize_array_parameter(self, name, value):
        value = tf.reshape(value, (-1,))
        # indices = tf.range(tf.size(value))

        # def fn(tuple):
        #     i, v = tuple
        #     self._summarize_scalar_parameter(f"{name}[{i}]", v)

        # tf.map_fn(fn, (indices, value))
        i = 0
        for v in value:
            print(v)
            self._summarize_scalar_parameter(f"{name}[{i}]", v)
            i += 1

    def _summarize_scalar_parameter(self, name, value):
        tf.summary.scalar(name, value, step=self.current_step)


class ScalarToTensorBoard(ToTensorBoard):
    """ Stores the returns value of a callback in a TensorBoard. """

    def __init__(self, log_dir: str, callback: Callable[[], float], name: str, period: int = 1):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param callback: callback to be executed and result written to TensorBoard.
        :param name: name used in TensorBoard.
        :param period: interval at which to run the task.
            For large values of`period` the task will be less frequently ran.
        """
        super().__init__(log_dir, period)
        self.name = name
        self.callback = callback

    def run(self, **kwargs):
        tf.summary.scalar(self.name, self.callback(**kwargs), step=self.current_step)


class ImageToTensorBoard(ToTensorBoard):
    def __init__(
        self,
        log_dir: str,
        plotting_function: Callable[[Figure, Axes], Figure],
        name: Optional[str] = None,
        period: int = 1,
        *,
        fig_kw: Optional[Dict[str, Any]] = None,
        subplots_kw: Optional[Dict[str, Any]] = None,
    ):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param plotting_function: function performing the plotting.
        :param name: name used in TensorBoard.
        :param period: interval at which to run the task.
            For large values of`period` the task will be less frequently ran.
        :params fig_kw: Keywords to be passed to Figure constructor, such as `figsize`.
        :params subplots_kw: Keywords to be passed to figure.subplots constructor, such as
            `nrows`, `ncols`, `sharex`, `sharey`. By default the default values 
            from matplotlib.pyplot as used.
        """
        super().__init__(log_dir, period)
        self.plotting_function = plotting_function
        self.name = name
        self.file_writer = tf.summary.create_file_writer(log_dir)
        self.name = name
        self.fig_kw = fig_kw or {}
        self.subplots_kw = subplots_kw or {}

        self.fig = Figure(**self.fig_kw)
        if self.subplots_kw != {}:
            self.axes = self.fig.subplots(**self.subplots_kw)
        else:
            self.axes = self.fig.add_subplot(111)

    def _clear_axes(self):
        try:
            for ax in self.axes.flatten():
                ax.clear()
        except (AttributeError, TypeError):
            self.axes.clear()

    def run(self, **unused_kwargs):
        self._clear_axes()
        self.plotting_function(self.fig, self.axes)
        canvas = FigureCanvasAgg(self.fig)
        canvas.draw()

        # get PNG data from the figure
        png_buffer = BytesIO()
        canvas.print_png(png_buffer)
        png_encoded = png_buffer.getvalue()
        png_buffer.close()

        image_tensor = tf.io.decode_png(png_encoded)[None]

        # Write to TensorBoard
        tf.summary.image(self.name, image_tensor, step=self.current_step)


class MonitorCollection:
    """
    Aggregates a list of tasks and allows to run them all with one call to `__call__`.
    """

    def __init__(self, tasks: List[MonitorTask]):
        """
        :param tasks: a list of `MonitorTask`s to be ran.
        """
        self.tasks = tasks

    def __call__(self, step, **kwargs):
        for task in self.tasks:
            task(step, **kwargs)

