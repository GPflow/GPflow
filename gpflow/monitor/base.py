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

""" MonitorTask base classes """

from abc import ABC, abstractmethod
from typing import Any, Callable, Collection, Union

import tensorflow as tf

__all__ = [
    "ExecuteCallback",
    "Monitor",
    "MonitorTask",
    "MonitorTaskGroup",
]


class MonitorTask(ABC):
    """
    A base class for a monitoring task.

    All monitoring tasks are callable objects.
    A descendant class must implement the `run` method, which is the body of the monitoring task.
    """

    def __call__(self, step: int, **kwargs: Any) -> None:
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
    def run(self, **kwargs: Any) -> None:
        """
        Implements the task to be executed on __call__.
        The current step is available through `self.current_step`.

        :param kwargs: keyword arguments available to the run method.
        """
        raise NotImplementedError


class ExecuteCallback(MonitorTask):
    """ Executes a callback as task """

    def __init__(self, callback: Callable[..., None]) -> None:
        """
        :param callback: callable to be executed during the task.
            Arguments can be passed using keyword arguments.
        """
        super().__init__()
        self.callback = callback

    def run(self, **kwargs: Any) -> None:
        self.callback(**kwargs)


class MonitorTaskGroup:
    """
    Class for grouping `MonitorTask` instances. A group defines
    all the tasks that are run at the same frequency, given by `period`.

    A `MonitorTaskGroup` can exist of a single instance or a list of
    `MonitorTask` instances.
    """

    def __init__(
        self, task_or_tasks: Union[Collection[MonitorTask], MonitorTask], period: int = 1
    ) -> None:
        """
        :param task_or_tasks: a single instance or a list of `MonitorTask` instances.
            Each `MonitorTask` in the list will be run with the given `period`.
        :param period: defines how often to run the tasks; they will execute every `period`th step.
            For large values of `period` the tasks will be less frequently run. Defaults to
            running at every step (`period = 1`).
        """
        self._tasks: Collection[MonitorTask] = []
        self.tasks = task_or_tasks  # type: ignore[assignment]
        self._period = period

    @property
    def tasks(self) -> Collection[MonitorTask]:
        return self._tasks

    @tasks.setter
    def tasks(self, task_or_tasks: Union[Collection[MonitorTask], MonitorTask]) -> None:
        """Ensures the tasks are stored as a list. Even if there is only a single task."""
        if isinstance(task_or_tasks, MonitorTask):
            self._tasks = [task_or_tasks]
        else:
            assert isinstance(task_or_tasks, Collection)
            self._tasks = list(task_or_tasks)

    def __call__(self, step: int, **kwargs: Any) -> None:
        """Call each task in the group."""
        if step % self._period == 0:
            for task in self.tasks:
                task(step, **kwargs)


class Monitor:
    r"""
    Accepts any number of of `MonitorTaskGroup` instances, and runs them
    according to their specified periodicity.

    Example use-case::

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
    """

    def __init__(self, *task_groups: MonitorTaskGroup) -> None:
        """
        :param task_groups: a list of `MonitorTaskGroup`s to be executed.
        """
        self.task_groups = task_groups

    def __call__(self, step: int, **kwargs: Any) -> None:
        for group in self.task_groups:
            group(step, **kwargs)
