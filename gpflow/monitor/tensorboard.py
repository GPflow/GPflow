# Copyright 2020 The GPflow Contributors. All Rights Reserved.
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

""" Tasks that write to TensorBoard """

from io import BytesIO
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Union

import numpy as np
import tensorflow as tf

from ..base import Parameter
from ..models import BayesianModel
from ..utilities import parameter_dict
from .base import MonitorTask

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib


__all__ = [
    "ImageToTensorBoard",
    "ModelToTensorBoard",
    "ScalarToTensorBoard",
    "ToTensorBoard",
]


class ToTensorBoard(MonitorTask):
    writers: Dict[str, tf.summary.SummaryWriter] = {}

    def __init__(self, log_dir: str) -> None:
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be nested, e.g. ./logs/my_run/
        """
        super().__init__()
        if log_dir not in self.writers:
            self.writers[log_dir] = tf.summary.create_file_writer(log_dir)
        self.file_writer = self.writers[log_dir]

    def __call__(self, step: int, **kwargs: Any) -> None:
        with self.file_writer.as_default():
            super().__call__(step, **kwargs)
        self.file_writer.flush()


class ModelToTensorBoard(ToTensorBoard):
    """
    Monitoring task that creates a sensible TensorBoard for a model.

    Monitors all the model's parameters for which their name matches with `keywords_to_monitor`.
    By default, "kernel" and "likelihood" are elements of `keywords_to_monitor`.

    Example::

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
    ) -> None:
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

    def run(self, **unused_kwargs: Any) -> None:
        for name, parameter in parameter_dict(self.model).items():
            # check if the parameter name matches any of the specified keywords
            if self.summarize_all or any(keyword in name for keyword in self.keywords_to_monitor):
                # keys are sometimes prepended with a character, which we strip
                name = name.lstrip(self.left_strip_character)
                self._summarize_parameter(name, parameter)

    def _summarize_parameter(self, name: str, param: Union[Parameter, tf.Variable]) -> None:
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
            # if there's only one element do not add a numbered suffix
            tf.summary.scalar(name, param[0], step=self.current_step)
        else:
            it = range(size) if self.max_size == -1 else range(min(size, self.max_size))
            for i in it:
                tf.summary.scalar(f"{name}[{i}]", param[i], step=self.current_step)


class ScalarToTensorBoard(ToTensorBoard):
    """Stores the return value of a callback in a TensorBoard."""

    def __init__(self, log_dir: str, callback: Callable[[], float], name: str) -> None:
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

    def run(self, **kwargs: Any) -> None:
        tf.summary.scalar(self.name, self.callback(**kwargs), step=self.current_step)


class ImageToTensorBoard(ToTensorBoard):
    def __init__(
        self,
        log_dir: str,
        plotting_function: Callable[
            ["matplotlib.figure.Figure", "matplotlib.figure.Axes"], "matplotlib.figure.Figure"
        ],
        name: Optional[str] = None,
        *,
        fig_kw: Optional[Dict[str, Any]] = None,
        subplots_kw: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        :param log_dir: directory in which to store the tensorboard files.
            Can be nested: for example, './logs/my_run/'.
        :param plotting_function: function performing the plotting.
        :param name: name used in TensorBoard.
        :params fig_kw: keyword arguments to be passed to Figure constructor, e.g. `figsize`.
        :params subplots_kw: keyword arguments to be passed to figure.subplots constructor, e.g.
            `nrows`, `ncols`, `sharex`, `sharey`. By default the default values
            from matplotlib.pyplot are used.
        """
        super().__init__(log_dir)
        self.plotting_function = plotting_function
        self.name = name
        self.fig_kw = fig_kw or {}
        self.subplots_kw = subplots_kw or {}

        try:
            from matplotlib.figure import Figure
        except ImportError:
            raise RuntimeError("ImageToTensorBoard requires the matplotlib package to be installed")

        self.fig = Figure(**self.fig_kw)
        if self.subplots_kw != {}:
            self.axes = self.fig.subplots(**self.subplots_kw)
        else:
            self.axes = self.fig.add_subplot(111)

    def _clear_axes(self) -> None:
        if isinstance(self.axes, np.ndarray):
            for ax in self.axes.flatten():
                ax.clear()
        else:
            self.axes.clear()

    def run(self, **unused_kwargs: Any) -> None:
        from matplotlib.backends.backend_agg import FigureCanvasAgg

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
