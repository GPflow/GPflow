# Copyright 2017 the GPflow authors.
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

"""
This module provides a simple framework for creating an optimisation monitor.

The framework consists of the following elements:

- Main monitor class 'Monitor'. This is a callable class that should be passed as the
  step_callback argument to the optimiser's minimize method.

- Monitor tasks. These are classes derived from the `MonitorTask`. The monitor should be given a
  list of tasks to execute at each optimiser's iteration and after the optimisation is completed.
  The tasks implemented in these module cover the most common monitoring requirements. Custom
  tasks can be created if needed. In particular one might want to create a new TensorBoard task.
  The `BaseTensorBoardTask` provides a convenient base class for such tasks.

- Monitor context. This is a property bag created by the monitor. At each iteration the monitor
  updates the data in the context before passing it to the tasks.

- Task conditions. These are callable objects that control execution of tasks. Each task can be
  assigned a condition, in which case it will run only if the condition is met. Although the
  conditions doesn't have to inherit from any particular base class it is convenient to derive
  a new condition from `GenericCondition` or one of its descendants.

- Utilities.

Below is an example of a monitored optimisation constructed from the framework's elements.

import numpy as np
import gpflow
import gpflow.training.monitor as mon

# Generate some input data
#
np.random.seed(0)
X = np.random.rand(10000, 1) * 10
Y = np.sin(X) + np.random.randn(*X.shape)

# Create the model
#
model = gpflow.models.SVGP(X, Y, gpflow.kernels.RBF(1), gpflow.likelihoods.Gaussian(),
                           Z=np.linspace(0, 10, 5)[:, None],
                           minibatch_size=100, name='SVGP')
model.likelihood.variance = 0.01

# Create the `global_step` tensor that the optimiser will use to indicate the current step number.
# Note that not all optimisers use this variable.
#
session = model.enquire_session()
global_step = mon.create_global_step(session)

# Create monitor tasks

# This task will print the optimisation timings every 10-th iteration
#
print_task = mon.PrintTimingsTask()\
    .with_name('print')\
    .with_condition(mon.PeriodicIterationCondition(10))\

# This task will save the Tensorflow session every 15-th iteration.
# The directory pointed to by `checkpoint_dir` must exist.
#
checkpoint_task = mon.CheckpointTask(checkpoint_dir="./model-saves")\
        .with_name('checkpoint')\
        .with_condition(mon.PeriodicIterationCondition(15))\

# This task will create a TensorFlow summary of the model for TensorBoard. It will run at
# every 100-th iteration. We also want to do this after the optimisation is finished.
# *** IMPORTANT ***
# Please make sure that if multiple LogdirWriters are used they are created with different
# locations (event file directory and file suffix). It is possible to share a writer between
# multiple tasks. But it is not possible to share event location between multiple writers.
#
with mon.LogdirWriter('./model-tensorboard') as writer:
    tensorboard_task = mon.ModelToTensorBoardTask(writer, model)\
        .with_name('tensorboard')\
        .with_condition(mon.PeriodicIterationCondition(100))\
        .with_exit_condition(True)

    monitor_tasks = [print_task, tensorboard_task, checkpoint_task]

    optimiser = gpflow.train.AdamOptimizer(0.01)

    # Create a monitor and run the optimiser providing the monitor as a callback function
    #
    with mon.Monitor(monitor_tasks, session, global_step, print_summary=True) as monitor:
        optimiser.minimize(model, step_callback=monitor, global_step=global_step)
"""

import time
import abc
from typing import Callable, List, Dict, Set, Optional, Iterator, Any, Tuple
import itertools
import logging
import math
from pathlib import PurePath
import io
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from gpflow import params_as_tensors_for
from gpflow import settings
from gpflow.models import Model
from gpflow.params import Parameter
from gpflow.training.scipy_optimizer import ScipyOptimizer


def get_hr_time() -> float:
    """
    Gets high resolution time. Mainly defined here for the convenience of unit testing.
    """
    return timer()


def create_global_step(session: tf.Session) -> tf.Variable:
    """
    Creates the Tensorflow 'global_step' variable (see `MonitorContext.global_step_tensor`).
    :param session: Tensorflow session the optimiser is running in
    :return: The variable tensor.
    """
    global_step_tensor = tf.Variable(0, trainable=False, name="global_step")
    session.run(global_step_tensor.initializer)
    return global_step_tensor


def restore_session(session: tf.Session, checkpoint_dir: str,
                    saver: Optional[tf.train.Saver] = None) -> None:
    """
    Restores Tensorflow session from the latest checkpoint.
    :param session: The TF session
    :param checkpoint_dir: checkpoint files directory.
    :param saver: The saver object, if not provided a default saver object will be created.
    """
    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    logger = settings.logger()
    if logger.isEnabledFor(logging.INFO):
        logger.info("Restoring session from `%s`.", checkpoint_path)

    saver = saver or get_default_saver()
    saver.restore(session, checkpoint_path)


def get_default_saver(max_to_keep: int=3) -> tf.train.Saver:
    """
    Creates Tensorflow Saver object with 3 recent checkpoints to keep.
    :param max_to_keep: Maximum number of recent checkpoints to keep, defaults to 3
    """
    return tf.train.Saver(max_to_keep=max_to_keep)


def update_optimiser(context, *args, **kwargs) -> None:
    """
    Writes optimiser state into corresponding TensorFlow variables. This may need to be done
    for optimisers like ScipyOptimiser that work with their own copies of the variables.
    Normally the source variables would be updated only when the optimiser has finished the
    minimisation. This function may be called from the callback in order to get the TensorFlow
    variables up-to-date so that they can be written into TensorBoard for example.

    The expected state parameters and the procedure of writing them into variables is specific
    to each optimiser. Currently it is implemented only for ScipyOptimiser.

    After the state is updated a flag is set to the context to prevent multiple updates in
    the same callback.

    :param context: Monitor context
    :param args: Optimiser's state passed to the callback
    :param kwargs: Optimiser's state passed to the callback
    """

    if context.optimiser is None or context.optimiser_updated:
        return

    if isinstance(context.optimiser, ScipyOptimizer) and len(args) > 0:

        optimizer = context.optimiser.optimizer  # get access to ExternalOptimizerInterface
        var_vals = [args[0][packing_slice] for packing_slice in optimizer._packing_slices]
        context.session.run(optimizer._var_updates,
                            feed_dict=dict(zip(optimizer._update_placeholders, var_vals)))
        context.optimiser_updated = True


class MonitorContext(object):
    """
    This is a property bag that will be passed to all monitoring tasks. New attributes can
    be added here when needed. This doesn't require changes in the monitor task interface.
    Below is the list of currently defined public attributes.

    - iteration_no: Current optimisation iteration number.

    - optimisation_time: Total time elapsed excluding monitoring tasks.

    - total_time: Total time elapsed including monitoring tasks.

    - optimisation_finished: Optimisation finished flag.

    - session: Tensorflow session the optimiser is running in.

    - global_step_tensor: 'global_step' Tensorflow variable. This is used by all (or most)
        Tensorflow optimisers to indicate the current step number.

    - init_global_step: Initial value of the global step. This will be checked at the start of
        monitoring. The value may be greater than zero if the graph was restored from a checkpoint.

    - optimiser: Optimiser running under the monitor

    - optimiser_updated: Flag indicating that the optimiser's state has already been written to
        the correspondent variables in the current step.
    """
    def __init__(self) -> None:

        self.iteration_no = 0
        self.optimisation_time = 0.0
        self.total_time = 0.0
        self.optimisation_finished = False
        self.session = None          # type: tf.Session
        self.global_step_tensor = None  # type: tf.Variable
        self.init_global_step = 0
        self.optimiser = None   # type: Any
        self.optimiser_updated = False

    @property
    def global_step(self) -> int:
        """
        Evaluates the value of the global step variable if it is set, otherwise returns the
        current iteration number.
        """
        if self.session is None or self.global_step_tensor is None:
            return self.iteration_no + self.init_global_step
        else:
            return self.session.run(self.global_step_tensor)


class MonitorTask(metaclass=abc.ABCMeta):
    """
    A base class for a monitoring task.
    All monitoring tasks are callable objects that keep track of their execution time.

    A descendant class must implement the `run` method, which is the body of the monitoring task.

    A task can be assigned a condition that will be evaluated before calling the `run` method.
    The condition is a function that takes a MonitorContext object as a parameter and returns a
    boolean value. The task will be executed only if the condition function returns True. The
    condition function will be called only if the optimisation is still running. A condition can
    be assigned using 'with_condition' function.

    If the task is called when the optimisation has already finished the execution of `run` will
    depend on the `exit condition`. This is a simple boolean flag which allows or disallows `run`
    after the optimisation is done. The flag can be set using `with_exit_condition` function.
    """

    def __init__(self, need_optimiser_update: Optional[bool]=False) -> None:
        """
        :param need_optimiser_update: Need to make sure the optimiser's state is written into
        corresponding TensorFlow variables every time the task is fired.
        """

        self._condition = lambda context: True
        self._exit_condition = False
        self._task_name = self.__class__.__name__
        self._total_time = 0.0
        self._last_call_time = 0.0
        self._need_optimiser_update = need_optimiser_update

    def with_condition(self, condition: Callable[[MonitorContext], bool]) -> 'MonitorTask':
        """
        Sets the task running condition that will be evaluated during the optimisation cycle.
        """
        self._condition = condition
        return self

    def with_exit_condition(self, exit_condition: Optional[bool]=True) -> 'MonitorTask':
        """
        Sets the flag indicating that the task should also run after the optimisation is ended.
        """
        self._exit_condition = exit_condition
        return self

    def with_name(self, task_name: str) -> 'MonitorTask':
        """
        Sets the task name. The name will be used when printing the execution time summary.
        By default the name is set to the tasks' class name.
        """
        self._task_name = task_name
        return self

    @property
    def task_name(self):
        """Task name"""
        return self._task_name

    @property
    def total_time(self):
        """Accumulated execution time of this task"""
        return self._total_time

    @property
    def last_call_time(self):
        """Last execution time"""
        return self._last_call_time

    @abc.abstractmethod
    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        """
        Monitoring task body which must be implemented by descendants.
        :param context: Monitor context.
        """
        raise NotImplementedError

    def __call__(self, context: MonitorContext, *args, **kwargs) -> None:
        """
        Class as a function implementation. It calls the 'run' function and measures its
        execution time.
        :param context: Monitor context.
        Extra arguments are the arguments passed by the optimiser in the callback function.
        """
        start_timestamp = get_hr_time()
        try:
            # Evaluate either the normal iteration loop condition or the exit condition
            # and run the task.
            fire_task = self._exit_condition if context.optimisation_finished else \
                self._condition(context)
            if fire_task:
                if self._need_optimiser_update:
                    update_optimiser(context, *args, **kwargs)
                self.run(context, *args, **kwargs)
        finally:
            # Remember the time of the last execution and update the accumulated time.
            self._last_call_time = get_hr_time() - start_timestamp
            self._total_time += self._last_call_time


class Monitor(object):
    """
    Main monitor class.
    It's main purpose is to provide a callback function that will be hooked onto an optimiser.

    In its initialisation it will create a MonitorContext object that will be passed to monitoring
    tasks.

    It is recommended to open the Monitor in a context using `with` statement (see the module-level
    doc).
    """

    def __init__(self, monitor_tasks: Iterator[MonitorTask], session: Optional[tf.Session]=None,
                 global_step_tensor: Optional[tf.Variable]=None,
                 print_summary: Optional[bool]=False,
                 optimiser: Optional[Any]=None, context: Optional[MonitorContext]=None) -> None:
        """
        :param monitor_tasks: A collection of monitoring tasks to run. The tasks will be called in
        the same order they are specified here.
        :param session: Tensorflow session the optimiser is running in.
        :param global_step_tensor: the Tensorflow 'global_step' variable
        (see notes in MonitorContext.global_step_tensor)
        :param print_summary: Prints tasks' timing summary after the monitoring is stopped.
        :param optimiser: Optimiser object that is going to run under this monitor.
        :param context: MonitorContext object, if not provided a new one will be created.
        """

        self._monitor_tasks = list(monitor_tasks)
        self._context = context or MonitorContext()
        self._context.optimisation_finished = False
        if session is not None:
            self._context.session = session
        if global_step_tensor is not None:
            self._context.global_step_tensor = global_step_tensor
        if optimiser is not None:
            self._context.optimiser = optimiser
        self._print_summary = print_summary

        self._start_timestamp = get_hr_time()
        self._last_timestamp = self._start_timestamp

    def __enter__(self):
        self.start_monitoring()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_monitoring()

    def __call__(self, *args, **kwargs) -> None:
        """
        Class as a function implementation. Normally it would be called by an optimiser when it
        completes an iteration. The optimiser may pass some arguments, e.g. the state of trainable
        variables. This arguments will be passed to all monitoring tasks in case they make sense
        to some of them.
        """
        self._on_iteration(*args, **kwargs)

    def start_monitoring(self) -> None:
        """
        The recommended way of using Monitor is opening it with the `with` statement. In this case
        the user doesn't need to call this function explicitly. Otherwise, the function should be
        called before starting the optimiser.

        The function evaluates the global_step variable in order to get its initial value. It also
        resets the starting timer since the time set in the __init__ may no longer be accurate.
        """
        self._context.init_global_step = self._context.global_step
        self._start_timestamp = get_hr_time()
        self._last_timestamp = self._start_timestamp

    def stop_monitoring(self) -> None:
        """
        The recommended way of using Monitor is opening it with the `with` statement. In this case
        the user doesn't need to call this function explicitly. Otherwise the function should be
        called when the optimisation is done.

        The function sets the optimisation completed flag in the monitoring context and runs the
        tasks once more. If the monitor was created with the `print_summary` option it prints the
        tasks' timing summary.
        """

        self._context.optimisation_finished = True
        self._on_iteration()

        if self._print_summary:
            self.print_summary()

    def print_summary(self) -> None:
        """
        Prints the tasks' timing summary.
        """
        print("Tasks execution time summary:")
        for mon_task in self._monitor_tasks:
            print("%s:\t%.4f (sec)" % (mon_task.task_name, mon_task.total_time))

    def _on_iteration(self, *args, **kwargs) -> None:
        """
        Called at each iteration.

        This function does time measurements, updates timing in the monitoring context and calls
        all monitoring tasks.
        """

        # Update timing and the iteration number in the monitoring context.
        current_timestamp = get_hr_time()
        self._context.optimisation_time += current_timestamp - self._last_timestamp
        self._context.total_time = current_timestamp - self._start_timestamp
        self._context.optimiser_updated = False
        if not self._context.optimisation_finished:
            self._context.iteration_no += 1

        # Call all monitoring functions
        for func in self._monitor_tasks:
            func(self._context, *args, **kwargs)

        # Remember the time when the control is returned back to the optimiser, so that the next
        # optimisation step can be accurately timed.
        self._last_timestamp = get_hr_time()


class GenericCondition(object):
    """
    This is a generic task running condition class.

    It is defined by a pair of objects:
    - Trigger function that selects or constructs a trigger from the monitoring context.
      The trigger value should be a strictly monotonically increasing function, like for example
      the iteration number. Other obvious options are the total time and optimisation time.

    - A generator that produces a sequence of trigger levels. The task will run if the trigger
      value has reached or exceeded the next level. The sequence should be infinite.
    """

    def __init__(self, trigger: Callable[[MonitorContext], Any], sequence: Iterator) -> None:
        """
        :param trigger: Trigger function
        :param sequence: Generator of the trigger levels
        """

        self._sequence = sequence
        self._trigger = trigger
        self._next = next(self._sequence)   # next level

    def __call__(self, context: MonitorContext) -> bool:

        trigger_value = self._trigger(context)
        if trigger_value >= self._next:
            # Move to the next trigger level, and make sure it's above the current trigger value
            while trigger_value >= self._next:
                self._next = next(self._sequence)
            return True
        else:
            return False


class PeriodicCondition(GenericCondition):
    """
    Condition that will run the task at equal intervals of the trigger value.
    """

    def __init__(self, trigger: Callable[[MonitorContext], Any], interval) -> None:
        """
        :param trigger: Trigger function as in the parent class
        :param interval: Trigger value interval
        """
        super().__init__(trigger, itertools.count(start=interval, step=interval))


class PeriodicIterationCondition(PeriodicCondition):
    """
    Specification of the periodic condition for the iteration number. It will run the task
    at every n-th iteration.
    """

    def __init__(self, interval: Optional[int]=1):
        """
        :param interval: Task running interval in the number of iterations, by default the task
        will run at every iteration.
        """
        super().__init__(lambda context: context.iteration_no, interval)


class GrowingIntervalCondition(GenericCondition):
    """
    Condition that will run the task at increasing intervals of the trigger value.
    """

    def __init__(self, trigger: Callable[[MonitorContext], Any], interval_growth: float,
                 max_interval, init_interval, start_value=None) -> None:
        """
        :param trigger: Trigger function as in the parent class
        :param interval_growth: Intervals will be increasing by this factor until they reach the
        specified maximum.
        :param max_interval: Maximum interval.
        :param init_interval: Initial interval.
        :param start_value: Trigger value when the task should run first time, defaults to the
        initial interval.
        """
        super().__init__(trigger, self._growing_step_sequence(interval_growth, max_interval,
                                                              init_interval, start_value))

    @staticmethod
    def _growing_step_sequence(interval_growth, max_interval, init_interval, start_level=None):
        """
        Returns an iterator that constructs a sequence of trigger levels with growing intervals.
        The interval is growing exponentially until it reaches the maximum value. Then the interval
        stays the same and the sequence becomes linear.

        An optional starting level `start_level` defaults to the initial interval. The interval
        starts out as `init_interval`, multiplied by `interval_growth` in each step until it
        reaches the `max_interval`.
        """
        interval = init_interval
        next_level = start_level or init_interval
        while True:
            yield next_level
            interval = min(interval * interval_growth, max_interval)
            next_level += interval


class PrintTimingsTask(MonitorTask):
    """
    Monitoring task that measures the optimisation speed and prints it to a file, or to sys.stdout
    by default. It measures the number of iterations per second and number of optimisation steps
    per second excluding the time spent on monitoring tasks/

    The following metrics are printed:
    - Overall iteration rate (iterations / second)
    - Recent iteration rate (iteration / second) since the last time this task was running.
    - Overall optimisation rate (optimisation steps / second)
    - Recent optimisation rate (optimisation steps / second) since the last time this task was
      running.

    The optimisation step may or may not be the same as the iteration number depending on the
    logic of a particular optimiser. Also, if the optimiser uses the Tensorflow global_step
    variable and the session is restored from a checkpoint then the optimisation step count will
    continue from the value stored in the checkpoint while the iteration count will start from zero.

    The output is in the form of Tab separated columns. If different format is desired then this
    task can be sub-classed and _print_timings function overridden.
    """

    def __init__(self, file=None) -> None:
        """
        :param file: Stream to print the output to.
        """
        super().__init__()
        self._file = file
        self._last_iter = 0
        self._last_step = 0
        self._last_time = 0.0
        self._last_time_opt = 0.0

    def run(self, context: MonitorContext, *args, **kwargs) -> None:

        global_step = context.global_step
        current_step = global_step - context.init_global_step

        # Iterations / per seconds
        total_iter_rate = context.iteration_no / context.total_time \
            if context.total_time > 0.0 else np.nan
        elapsed_time = context.total_time - self._last_time
        recent_iter_rate = (context.iteration_no - self._last_iter) / elapsed_time \
            if elapsed_time > 0.0 else np.nan

        # Optimisation steps / seconds (excluding the time spent on monitoring tasks)
        total_opt_rate = current_step / context.optimisation_time \
            if context.optimisation_time > 0.0 else np.nan
        elapsed_time_opt = context.optimisation_time - self._last_time_opt
        recent_opt_rate = (current_step - self._last_step) / elapsed_time_opt \
            if elapsed_time_opt > 0.0 else np.nan

        self._last_iter = context.iteration_no
        self._last_step = current_step
        self._last_time = context.total_time
        self._last_time_opt = context.optimisation_time

        self._print_timings(context.iteration_no, global_step, total_iter_rate, recent_iter_rate,
                            total_opt_rate, recent_opt_rate)

    def _print_timings(self, iter_no: int, step_no: int,
                       total_iter_rate: float, recent_iter_rate: float,
                       total_opt_rate: float, recent_opt_rate: float) -> None:

        print("Iteration %i\ttotal itr.rate %.2f/s\trecent itr.rate %.2f/s"
              "\topt.step %i\ttotal opt.rate %.2f/s\trecent opt.rate %.2f/s" %
              (iter_no, total_iter_rate, recent_iter_rate, step_no, total_opt_rate,
               recent_opt_rate), file=self._file)


class CallbackTask(MonitorTask):
    """
    Monitoring task that allows chaining monitors.

    This can be useful if several tasks need to run under the same condition. Instead of
    creating multiple replicas of this condition the tasks can be put into a separate monitor.
    This monitor will be linked to the main one through the callback task and the condition
    will be assigned to it.
    """

    def __init__(self, callback: Callable) -> None:

        super().__init__()
        self._callback = callback

    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        self._callback(context, *args, **kwargs)


class SleepTask(MonitorTask):
    """
    Monitoring task that slows down the optimiser.
    This can be used to observe the progress of the optimisation.
    """

    def __init__(self, sleep_seconds: float) -> None:
        """
        :param sleep_seconds: Sleep interval (seconds) between subsequent iterations.
        """
        super().__init__()
        self._sleep_seconds = sleep_seconds

    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        time.sleep(self._sleep_seconds)


class CheckpointTask(MonitorTask):
    """
    Monitoring task that creates a checkpoint using the Tensorflow Saver.
    """

    def __init__(self, checkpoint_dir: str, saver: Optional[tf.train.Saver] = None,
                 checkpoint_file_prefix: Optional[str]='cp') -> None:
        """
        :param checkpoint_dir: Directory where the checkpoint files will be created.
        The directory must exist.
        :param saver: The saver object. If not specified a default Saver will be created.
        :param checkpoint_file_prefix: Prefix of file names created for the checkpoint.
        """
        super().__init__()

        self._checkpoint_path = str(PurePath(checkpoint_dir, checkpoint_file_prefix))
        self._saver = saver or get_default_saver()

    def run(self, context: MonitorContext, *args, **kwargs) -> None:

        self._saver.save(context.session, self._checkpoint_path,
                         global_step=context.global_step_tensor)


class LogdirWriter(tf.summary.FileWriter):
    """
    This is a wrapper around the TensorFlow summary.EventWriter that provides a workaround for
    a bug currently present in this module. The EventWriter can only open the file in exclusive
    mode, however when multiple instances of the writer attempt to access the same file no error
    is raised.

    This class prevents user from opening multiple writers with the same location (event file
    directory and file name suffix). It keeps a global set of used locations adding a new location
    there when a writer is created or reopened. It removes the location from the global set when
    the writer is closed or garbage collected.

    Once the bug in TensorFlow is fixed this class can be removed or reduced to trivial:
    class LogdirWriter(tf.summary.FileWriter):
        pass
    """

    _locked_locations = set()     # type: Set[Tuple[str, Optional[str]]

    def __init__(self, logdir: str, graph: Optional[tf.Graph]=None, max_queue: int=10,
                 flush_secs: float=120, filename_suffix: Optional[str]=None):
        """
        A thin wrapper around the summary.FileWriter __init__. It remembers the location (a tuple
        of event file directory and file name suffix) and attempts to lock it. If the location
        is already locked by another writer an error will be raised.

        :param logdir: Directory where event file will be written.
        :param graph: A `Graph` object
        :param max_queue: Size of the queue for pending events and summaries.
        :param flush_secs: How often, in seconds, to flush the added summaries and events to disk.
        :param filename_suffix: Optional suffix of the event file's name.
        """

        self._location = (str(PurePath(logdir)), filename_suffix)
        self._is_active = False
        self.__lock_location()
        super().__init__(logdir, graph, max_queue, flush_secs, filename_suffix=filename_suffix)

    def __del__(self):
        self.__release_location()
        if hasattr(super(), '__del__'):
            super().__del__()

    def close(self) -> None:
        """
        Closes the summary.FileWriter. Releases the lock on the location so that another writer
        can take it.
        """
        super().close()
        self.__release_location()

    def reopen(self) -> None:
        """
        Reopens the summary.FileWriter that has been previously closed. Attempts to lock the event
        file location. Will raise an error if the location  has been taken by another writer since
        it was closed by this writer.
        """
        self.__lock_location()
        super().reopen()

    def __lock_location(self) -> None:
        """
        Attempts to lock the location used by this writer. Will raise an error if the location is
        already locked by another writer. Will do nothing if the location is already locked by
        this writer.
        """
        if not self._is_active:
            if self._location in LogdirWriter._locked_locations:
                raise RuntimeError('TensorBoard event file in directory %s with suffix %s '
                                   'is already in use. At present multiple TensoBoard file writers '
                                   'cannot write data into the same file.' % self._location)
            LogdirWriter._locked_locations.add(self._location)
            self._is_active = True

    def __release_location(self) -> None:
        """
        Releases the lock on the location used by this writer. Will do nothing if the lock is
        already released.
        """
        if self._is_active:
            LogdirWriter._locked_locations.remove(self._location)
            self._is_active = False


class BaseTensorBoardTask(MonitorTask):
    """
    Base class for TensorBoard monitoring tasks.

    A descendant task should create the Tensorflow Summary object in the __init__.
    If the summary object contains one or more placeholders it may also override the `run`
    method where it can calculate the correspondent values. It will then call the _eval_summary
    providing these values as the input values dictionary.

    A TensorBoard task requests access to the TensorFlow summary FileWriter object providing the
    location of the event file. The FileWriter object will be created if it doesn't exist. When
    the task is no longer needed the `close` method should be called. This will release the
    FileWriter object.
    """

    def __init__(self, file_writer: LogdirWriter, model: Optional[Model]=None) -> None:
        """
        :param file_writer: Event file writer object.
        :param model: Model object
        """
        super().__init__(need_optimiser_update=True)
        if not isinstance(file_writer, LogdirWriter):
            raise RuntimeError('The event file writer object provided to a TensorBoard task must '
                               'be of the type LogdirWriter or a descendant type.')
        self._file_writer = file_writer
        self._model = model
        self._summary = None    # type: tf.Summary
        self._flush_immediately = False

    @property
    def model(self) -> Model:
        """
        Model object
        """
        return self._model

    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        self._eval_summary(context)

    def with_flush_immediately(self, flush_immediately: Optional[bool]=True)\
            -> 'BaseTensorBoardTask':
        """
        Sets the flag indicating that the event file should be flushed at each call.
        """
        self._flush_immediately = flush_immediately
        return self

    def flush(self):
        """
        Flushes the event file to disk.
        This can be called to make sure that all pending events have been written to disk.
        """
        self._file_writer.flush()

    def _eval_summary(self, context: MonitorContext, feed_dict: Optional[Dict]=None) -> None:
        """
        Evaluates the summary tensor and writes the result to the event file.
        :param context: Monitor context
        :param feed_dict: Input values dictionary to be provided to the `session.run`
        when evaluating the summary tensor.
        """

        if self._summary is None:
            raise RuntimeError('TensorBoard monitor task should set the Tensorflow.Summary object')

        if context.session is None:
            raise RuntimeError('To run a TensorBoard monitor task the TF session object'
                               ' must be provided when creating an instance of the Monitor')

        summary = context.session.run(self._summary, feed_dict=feed_dict)
        self._file_writer.add_summary(summary, context.global_step)
        if self._flush_immediately:
            self.flush()


class ModelToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a sensible TensorBoard for a model.
    It sends to the TensorBoard the likelihood value and the model parameters.

    The user can specify a list of model parameters that should be sent to the TensorBoard. By
    default all model parameters are sent. The user can choose to send only scalar parameters.

    It is possible to provide an additional list of Tensorflow summary objects that will be
    merged with the model parameters' summary.
    """

    def __init__(self, file_writer: LogdirWriter, model: Model, only_scalars: bool = True,
                 parameters: Optional[List[Parameter]] = None,
                 additional_summaries: Optional[List[tf.Summary]] = None) -> None:
        """
        :param model: Model tensor
        :param file_writer: Event file writer object.
        :param only_scalars: Restricts the list of output parameters to scalars.
        :param parameters: List of model parameters to send to TensorBoard. If not
        provided all parameters will be sent to TensorBoard.
        :param additional_summaries: List of additional summary objects to send to TensorBoard.
        """
        super().__init__(file_writer, model)
        all_summaries = additional_summaries or []
        parameters = parameters or list(model.parameters)

        # Add scalar parameters
        all_summaries += [tf.summary.scalar(p.pathname, tf.reshape(p.constrained_tensor, []))
                          for p in parameters if p.size == 1]

        # Add non-scalar parameters
        if not only_scalars:
            all_summaries += [tf.summary.histogram(p.full_name, p.constrained_tensor)
                              for p in parameters if p.size > 1]

        # Add likelihood
        all_summaries.append(tf.summary.scalar("optimisation/likelihood",
                                               model._likelihood_tensor))

        # Create the summary tensor
        self._summary = tf.summary.merge(all_summaries)


class LmlToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with just one scalar value -
    the unbiased estimator of the evidence lower bound (ELBO or LML).

    The LML is averaged over a number of minimatches. The input dataset is split into the specified
    number of sequential minibatches such that every datapoint is used exactly once. The set of
    minibatches doesn't change from one iteration to another.

    The task can display the progress of calculating LML at each iteration (how many minibatches
    are left to compute). For that the `tqdm' progress bar should be installed (pip install tqdm).
    """

    def __init__(self, file_writer: LogdirWriter, model: Model, minibatch_size: Optional[int] = 100,
                 display_progress: Optional[bool] = True) -> None:
        """
        :param model: Model tensor
        :param file_writer: Event file writer object.
        :param minibatch_size: Number of points per minibatch
        :param display_progress: if True the task displays the progress of calculating LML.
        """

        super().__init__(file_writer, model)
        self._minibatch_size = minibatch_size
        self._full_lml = tf.placeholder(settings.tf_float, shape=())
        self._summary = tf.summary.scalar(model.name + '/full_lml', self._full_lml)

        self.wrapper = None  # type: Callable[[Iterator], Iterator]
        if display_progress:  # pragma: no cover
            try:
                import tqdm
                self.wrapper = tqdm.tqdm
            except ImportError:
                logger = settings.logger()
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning("LML monitor task: to display progress install `tqdm`.")
        if self.wrapper is None:
            self.wrapper = lambda x: x

    def run(self, context: MonitorContext, *args, **kwargs) -> None:

        with params_as_tensors_for(self._model):
            tf_x, tf_y = self._model.X, self._model.Y

        lml = 0.0
        num_batches =  int(math.ceil(len(self._model.X._value) / self._minibatch_size))  # round up
        for mb in self.wrapper(range(num_batches)):
            start = mb * self._minibatch_size
            finish = (mb + 1) * self._minibatch_size
            x_mb = self._model.X._value[start:finish, :]
            y_mb = self._model.Y._value[start:finish, :]
            mb_lml = self._model.compute_log_likelihood(feed_dict={tf_x: x_mb, tf_y: y_mb})
            lml += mb_lml * len(x_mb)
        lml = lml / len(self._model.X._value)

        self._eval_summary(context, {self._full_lml: lml})


class ScalarFuncToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with a single scalar value computed by a user
    provided function.
    """

    def __init__(self, file_writer: LogdirWriter, func: Callable, func_name: str) -> None:
        """
        :param file_writer: Event file writer object.
        :param func: User function that provides a value for the TensorBoard
        :param func_name: Name the function should be seen with in the TensorBoard. This name may
        get altered by tf.summary. For example spaces will be replaced with underscores.
        """
        super().__init__(file_writer)
        self.func = func
        self.placeholder = tf.placeholder(tf.float64)
        self._summary = tf.summary.scalar(func_name, self.placeholder)

    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        self._eval_summary(context, {self.placeholder: self.func(*args, **kwargs)})


class VectorFuncToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with multiple values computed by a user
    provided function. The function can return values in an array of various complexity.
    The array will be stored in the TensorBoard in a flat form made by the numpy `array.flatten()`
    function.
    """

    def __init__(self, file_writer: LogdirWriter, func: Callable, func_name: str,
                 num_outputs: int) -> None:
        """
        :param file_writer: Event file writer object.
        :param func: User function that provides vector values for the TensorBoard.
        :param func_name: Name the function should be seen with in the TensorBoard. This name may
        get altered by tf.summary. For example spaces will be replaced with underscores.
        :param num_outputs: The total number of values returned by the function.
        """

        super().__init__(file_writer)
        self.func = func
        self.placeholders = [tf.placeholder(tf.float64) for _ in range(num_outputs)]
        self._summary = tf.summary.merge([tf.summary.scalar(
            func_name + "_" + str(i), pl) for i, pl in enumerate(self.placeholders)])

    def run(self, context: MonitorContext, *args, **kwargs) -> None:

        values = np.array(self.func()).flatten()
        feeds = {pl: value for pl, value in zip(self.placeholders, values)}
        self._eval_summary(context, feeds)


class HistogramToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with a histogram made of values computed by a user
    provided function.
    """

    def __init__(self, file_writer: LogdirWriter, func: Callable, func_name: str, output_dims):
        """
        :param file_writer: Event file writer object.
        :param func: User function that provides histogram values for the TensorBoard.
        :param func_name: Name the function should be seen with in the TensorBoard. This name may
        get altered by tf.summary. For example spaces will be replaced with underscores.
        :param output_dims: The shape of the data returned by the function. The shape must be in
        the format accepted by the `tf.placeholder(...)`
        """

        super().__init__(file_writer)
        self.func = func
        self.placeholder = tf.placeholder(tf.float64, shape=output_dims)
        self._summary = tf.summary.histogram(func_name, self.placeholder)

    def run(self, context: MonitorContext, *args, **kwargs) -> None:
        self._eval_summary(context, {self.placeholder: self.func()})


class ImageToTensorBoardTask(BaseTensorBoardTask):
    """
    Monitoring task that creates a TensorBoard with an image returned by a user provided function.
    """

    def __init__(self, file_writer: LogdirWriter, func: Callable[[], Figure], func_name: str):
        """
        :param file_writer: Event file writer object.
        :param func: User function that provides histogram values for the TensorBoard.
        :param func_name: Name the function should be seen with in the TensorBoard. The name will
        be appended by '/image/0'. Th name itself may also get altered by tf.summary. For example
        spaces will be replaced with underscores.
        """

        super().__init__(file_writer)
        self.func = func
        self.placeholder = tf.placeholder(tf.float64, [1, None, None, None])
        self._summary = tf.summary.image(func_name, self.placeholder)

    def run(self, context: MonitorContext, *args, **kwargs) -> None:

        # Get the image and write it into a buffer in the PNG format.
        fig = self.func()
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        plt.close(fig)

        # Create TF image and load its content from the buffer.
        buf.seek(0)
        image = tf.image.decode_png(buf.getvalue())
        # Add the image number as a new dimension
        image = context.session.run(tf.expand_dims(image, 0))

        self._eval_summary(context, {self.placeholder: image})
