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

from .base import Parameter
from .models import BayesianModel
from .utilities import parameter_dict


class MonitorTask(ABC):
    """
    A base class for a monitoring task.

    All monitoring tasks are callable objects.
    A descendant class must implement the `run` method, which is the body of the monitoring task.
    """

    def __call__(self, step: int, **kwargs):
        """
        It calls the 'run' function and sets the current step.

        :param step: current step in the optimisation.
        :param kwargs: additional keyword arguments that can be passed
            to the `run` method of the task. This is in particular handy for
            passing keyword argument to the callback of `ScalarToTensorBoard`.
        """
        self.current_step = tf.cast(step, tf.int64)
        self.run(**kwargs)

    @abstractmethod
    def run(self, **kwargs):
        """
        Implements the task to be executed on __call__.
        The current step is available through `self.current_step`.

        :param kwargs: keyword arguments available to the run method.
        """
        raise NotImplementedError


class ExecuteCallback(MonitorTask):
    """ Executes a callback as task """

    def __init__(self, callback: Callable[..., None]):
        """
        :param callback: callable to be executed during the task.
            Arguments can be passed using keyword arguments.
        """
        super().__init__()
        self.callback = callback

    def run(self, **kwargs):
        self.callback(**kwargs)


# pylint: disable=abstract-method
class ToTensorBoard(MonitorTask):
    def __init__(self, log_dir: str):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be nested, e.g. ./logs/my_run/
        """
        super().__init__()
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def __call__(self, step, **kwargs):
        with self.file_writer.as_default():
            super().__call__(step, **kwargs)
        self.file_writer.flush()


class ModelToTensorBoard(ToTensorBoard):
    """
    Monitoring task that creates a sensible TensorBoard for a model.

    Monitors all the model's parameters for which their name matches with `keywords_to_monitor`.
    By default, "kernel" and "likelihood" are elements of `keywords_to_monitor`.
    Example:
        keyword = "kernel", parameter = "kernel.lengthscale" => match
        keyword = "variational", parameter = "kernel.lengthscale" => no match
    """

    def __init__(
        self,
        log_dir: str,
        model: BayesianModel,
        *,
        max_size: int = 3,
        keywords_to_monitor: List[str] = ["kernel", "likelihood"],
        left_strip_character: str = ".",
    ):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param model: model to be monitord.
        :param max_size: maximum size of arrays (incl.) to store each
            element of the array independently as a scalar in the TensorBoard.
            Setting max_size to -1 will write all values. Use with care.
        :param keywords_to_monitor: specifies keywords to be monitored.
            If the parameter's name includes any of the keywords specified it
            will be monitored. By default, parameters that match the `kernel` or
            `likelihood` keyword are monitored.
            Adding a "*" to the list will match with all parameters,
            i.e. no parameters or variables will be filtered out.
        :param left_strip_character: certain frameworks prepend their variables with
            a character. GPflow adds a '.' and Keras add a '_', for example.
            When a `left_strip_character` is specified it will be stripped from the
            parameter's name. By default the '.' is left stripped, for example:
            ".likelihood.variance" becomes "likelihood.variance".
        """
        super().__init__(log_dir)
        self.model = model
        self.max_size = max_size
        self.keywords_to_monitor = keywords_to_monitor
        self.summarize_all = "*" in self.keywords_to_monitor
        self.left_strip_character = left_strip_character

    def run(self, **unused_kwargs):
        for name, parameter in parameter_dict(self.model).items():
            # check if the parameter name matches any of the specified keywords
            if self.summarize_all or any(keyword in name for keyword in self.keywords_to_monitor):
                # keys are sometimes prepended with a character, which we strip
                name = name.lstrip(self.left_strip_character)
                self._summarize_parameter(name, parameter)

    def _summarize_parameter(self, name: str, param: Union[Parameter, tf.Variable]):
        """
        :param name: identifier used in tensorboard
        :param param: parameter to be stored in tensorboard
        """
        param = tf.reshape(param, (-1,))
        size = param.shape[0]

        if not isinstance(size, int):
            raise ValueError(
                f"The monitoring can not be autographed as the size of a parameter {param} "
                "is unknown at compile time. If compiling the monitor task is important, "
                "make sure the shape of all parameters is known beforehand. Otherwise, "
                "run the monitor outside the `tf.function`."
            )

        if size == 1:
            tf.summary.scalar(name, param[0], step=self.current_step)
        else:
            for i in range(min(size, self.max_size)):
                tf.summary.scalar(f"{name}[{i}]", param[i], step=self.current_step)


class ScalarToTensorBoard(ToTensorBoard):
    """Stores the return value of a callback in a TensorBoard."""

    def __init__(self, log_dir: str, callback: Callable[[], float], name: str):
        """
        :param log_dir: directory in which to store the tensorboard files.
            For example, './logs/my_run/'.
        :param callback: callback to be executed and result written to TensorBoard.
            A callback can have arguments (e.g. data) passed to the function using
            keyword arguments.
            For example:
            ```
            lambda cb(x=None): 2 * x
            task = ScalarToTensorBoard(logdir, cb, "callback")

            # specify the argument of the function using kwargs, the names need to match.
            task(step, x=1)
            ```
        :param name: name used in TensorBoard.
        """
        super().__init__(log_dir)
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
        *,
        fig_kw: Optional[Dict[str, Any]] = None,
        subplots_kw: Optional[Dict[str, Any]] = None,
    ):
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be a nested: for example, './logs/my_run/'.
        :param plotting_function: function performing the plotting.
        :param name: name used in TensorBoard.
        :params fig_kw: Keywords to be passed to Figure constructor, such as `figsize`.
        :params subplots_kw: Keywords to be passed to figure.subplots constructor, such as
            `nrows`, `ncols`, `sharex`, `sharey`. By default the default values 
            from matplotlib.pyplot as used.
        """
        super().__init__(log_dir)
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
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flatten():
                ax.clear()
        else:
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


class MonitorTaskGroup:
    """
    Class for grouping `MonitorTask` instances. A group defines
    all the tasks that are run at the same frequency, given by `period`.

    A `MonitorTaskGroup` can exist of a single instance or a list of
    `MonitorTask` instances.
    """

    def __init__(self, task_or_tasks: Union[List[MonitorTask], MonitorTask], period: int = 1):
        """
        :param task_or_tasks: a single instance or a list of `MonitorTask` instances.
            Each `MonitorTask` in the list will be run with the given `period`.
        :param period: defines how often to run the tasks; they will execute every `period`th step.
            For large values of `period` the tasks will be less frequently run. Defaults to
            running at every step (`period = 1`).
        """
        self.tasks = task_or_tasks
        self._period = period

    @property
    def tasks(self) -> List[MonitorTask]:
        return self._tasks

    @tasks.setter
    def tasks(self, task_or_tasks: Union[List[MonitorTask], MonitorTask]) -> None:
        """Ensures the tasks are stored as a list. Even if there is only a single task."""
        if not isinstance(task_or_tasks, List):
            self._tasks = [task_or_tasks]
        else:
            self._tasks = task_or_tasks

    def __call__(self, step, **kwargs):
        """Call each task in the group."""
        if step % self._period == 0:
            for task in self.tasks:
                task(step, **kwargs)


class Monitor:
    r"""
    Accepts any number of of `MonitorTaskGroup` instances, and runs them
    according to their specified periodicity.

    Example use-case:
        ```
        # Create some monitor tasks
        log_dir = "logs"
        model_task = ModelToTensorBoard(log_dir, model)
        image_task = ImageToTensorBoard(log_dir, plot_prediction, "image_samples")
        lml_task = ScalarToTensorBoard(log_dir, lambda: model.log_marginal_likelihood(), "lml")

        # Plotting tasks can be quite slow, so we want to run them less frequently.
        # We group them in a `MonitorTaskGroup` and set the period to 5.
        slow_tasks = MonitorTaskGroup(image_task, period=5)

        # The other tasks are fast. We run them at each iteration of the optimisation.
        fast_tasks = MonitorTaskGroup([model_task, lml_task], period=1)

        # We pass both groups to the `Monitor`
        monitor = Monitor(fast_tasks, slow_tasks)
        ```
    """

    def __init__(self, *task_groups: MonitorTaskGroup):
        """
        :param task_groups: a list of `MonitorTaskGroup`s to be executed.
        """
        self.task_groups = task_groups

    def __call__(self, step, **kwargs):
        for group in self.task_groups:
            group(step, **kwargs)
