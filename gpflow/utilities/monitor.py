from typing import Callable, Optional, Union
import collections

import numpy as np
import tensorflow as tf
from matplotlib.figure import Figure
from .utilities import parameter_dict

from ..models import BayesianModel

# fig = matplotlib.figure.Figure()   # or just `fig = tfplot.Figure()`
# ax = fig.add_subplot(1, 1, 1)      # ax: AxesSubplot


class MonitorTask:
    def __init__(self, period: int = 1):
        """
        :param period: int
            Interval between triggering the task.
        """
        self.period = period

    def __call__(self, step: int):
        if step % self.period == 0:
            self.current_step = step
            self.call()

    def call(self):
        raise NotImplementedError


class TensorBoardTask(MonitorTask):
    def __init__(self, log_dir: str, period: int = 1):
        super().__init__(period)
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def call(self):
        raise NotImplementedError


class ModelToTensorBoardTask(TensorBoardTask):
    def __init__(
        self, model: BayesianModel, log_dir: str, max_size: int = 3, period: int = 1
    ):
        """
        :param model:
        :param log_dir:
        :param max_size:
        :param period:
        """
        super().__init__(log_dir, period)
        self.model = model
        self.max_size = max_size

    def call(self):
        for k, v in parameter_dict(self.model).items():
            self._summarize_parameter(k, v.numpy())

    def _summarize_parameter(self, name: str, value: Union[float, np.ndarray]):
        """
        :param name: str
        :param value
        """
        name = name.lstrip(".")
        if isinstance(value, float):
            self._summarize_scalar_parameter(name, value)
        elif isinstance(value, np.ndarray) and value.size <= self.max_size:
            print("dkjfhsdlkfjsldkfJ")
            self._summarize_array_parameter(name, value)
        else:
            print(value.size)
            print("here")

    def _summarize_scalar_parameter(self, name, value):
        tf.summary.scalar(name, value, step=self.current_step)
        print(name, self.current_step, value)

    def _summarize_array_parameter(self, name, value):
        for i, v in enumerate(value.flatten()):
            tf.summary.scalar(f"name[{i}]", v, step=self.current_step)
            print(f"{name}[{i}], {v}, {self.current_step}")


class ScalarToTensorBoard(TensorBoardTask):
    def __init__(
        self, function: Callable[[], float], name: str, log_dir: str, period: int = 1
    ):
        super().__init__(log_dir, period)
        self.name = name
        self.function = function

    def call(self, step: int):
        tf.summary.scalar(self.name, self.function(), step=step)


# class ImageToTensorBoardTask(MonitorTask):

#     def __init__(
#         self,
#         log_dir: str,
#         plotting_function: Callable[[Figure, AxesSubplot], Figure],
#         name: Optional[str] = None
#     ):
#         """
#         :param file_writer: Event file writer object.
#         :param plotting_function: Callable that returns a figure
#         :param name: Name the function should be seen with in the TensorBoard. The name will
#         be appended by '/image/0'. Th name itself may also get altered by tf.summary. For example
#         spaces will be replaced with underscores.
#         """
#         super().__init__()
#         self.plotting_function = plotting_function
#         self.name = name
#         self.file_writer = tf.summary.create_file_writer(log_dir)
