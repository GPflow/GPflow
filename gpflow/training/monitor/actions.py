import enum
import glob
import os
import time
from typing import Any, Callable, List, Optional, Iterator
from typing import Optional

import numpy as np
import tensorflow as tf

from ...actions import Action, ActionContext, Watcher
from ...models import Model
from ...params import Parameter


# TODO: Make TriggeredAction, which allows a sequence to be passed of iterations or times for the action to be run.
# TODO: Implement TensorBoard action
# TODO: Implement model saving action
# TODO: Make PrintAction print the timings, like before
# TODO: Change interface to have sequence and trigger do tasks every iteration by default


@enum.unique
class Trigger(enum.Enum):
    TOTAL_TIME = 1
    OPTIMISATION_TIME = 2
    ITER = 3


class TriggeredAction(Action):
    def __init__(self, sequence: Iterator, trigger: Trigger) -> None:
        super().__init__()
        self._seq = sequence
        self.trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf

    def _current_trigger_value(self, context: ActionContext):
        if self.trigger == Trigger.TOTAL_TIME:
            return context.time_spent
        elif self.trigger == Trigger.OPTIMISATION_TIME:
            raise NotImplementedError
        elif self.trigger == Trigger.ITER:
            return context.iteration
        else:
            raise NotImplementedError

    def __call__(self, context: Optional[ActionContext] = None) -> None:
        """
        Action call method.
        The `action()` activates watcher, then calls `run` method and deactivates
        watcher even when `run` method raised an exception.

            :param context: Optional action context value. When argument is none
                the standard ActionContext is created and callee action assigns
                itself as a context owner.
        """
        context = ActionContext(self) if context is None else context
        if self._current_trigger_value(context) > self._next:
            try:
                self.watcher.start()
                self.run(context)
            finally:
                self.watcher.stop()

            # Move to the next trigger time, and make sure it's after this current iteration
            while self._next < self._current_trigger_value(context):
                self._next = next(self._seq)


class PrintLikelihood(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger, model: Model, text: str, *,
            single_line: bool = False) -> None:
        super().__init__(sequence, trigger)
        self.model = model
        self.text = text
        self.single_line = single_line

    def run(self, ctx: ActionContext):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood),
              end="\r" if self.single_line else "\n")


class PrintTimings(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger,
            global_step: Optional[tf.Variable] = None, single_line: bool = True) -> None:
        super().__init__(sequence, trigger)
        self.global_step = global_step
        self.single_line = single_line

    def run(self, context: ActionContext) -> None:
        current_iter = context.iteration
        if current_iter == 0:
            opt_iter = 0.0
            total_iter = 0.0
            last_iter = 0.0
        else:
            opt_iter = np.nan
            total_iter = current_iter / context.time_spent
            last_iter = (0.0 if not hasattr(self, '_last_iter')
                         else (current_iter - self._last_iter) / self._last_iter_timer.elapsed)

        step = context.iteration if self.global_step is None else context.session.run(self.global_step)
        print("\r%i, %i:\t%.2f optimisation iter/s\t%.2f total iter/s\t%.2f last iter/s" %
              (current_iter, step, opt_iter, total_iter, last_iter), end='' if self.single_line else '\n')

        self._last_iter = current_iter
        self._last_iter_timer = Watcher()
        self._last_iter_timer.start()


class CallbackAction(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger,
                 callback: Callable, model: Model, **kwargs) -> None:
        super().__init__(sequence, trigger)
        self._callback = lambda ctx: callback(ctx, model, **kwargs)

    def run(self, ctx: ActionContext):
        self._callback(ctx)


class SleepAction(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger, sleep_seconds: float) -> None:
        super().__init__(sequence, trigger)
        self.sleep_seconds = sleep_seconds

    def run(self, ctx):
        time.sleep(self.sleep_seconds)


class StoreSession(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger, session: tf.Session, hist_path: str,
            saver: Optional[tf.train.Saver] = None, restore_path: Optional[str] = None,
            global_step: Optional[tf.Variable] = None) -> None:
        super().__init__(sequence, trigger)
        self.hist_path = hist_path
        self.restore_path = restore_path
        self.saver = tf.train.Saver(max_to_keep=3) if saver is None else saver
        self.session = session
        self.global_step = global_step

        restore_path = self.restore_path
        if restore_path is None:
            if len(glob.glob(self.hist_path + "-*")) > 0:
                # History exists
                latest_step = np.max([int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
                                      for x in glob.glob(self.hist_path + "-*")])
                restore_path = self.hist_path + "-%i" % latest_step

        if restore_path is not None:
            print("Restoring session from `%s`." % restore_path)
            self.saver.restore(session, restore_path)

    def run(self, ctx: ActionContext):
        self.saver.save(self.session, self.hist_path, global_step=self.global_step)


class ModelTensorBoard(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger, model: Model, file_writer: tf.summary.FileWriter,
            only_scalars: bool = True,
            parameters: Optional[List[Parameter]] = None,
            additional_summaries: Optional[List[tf.Summary]] = None,
            global_step: Optional[tf.Variable] = None):
        """
        Creates a Task that creates a sensible TensorBoard for a model.
        :param sequence:
        :param trigger:
        :param model:
        :param file_writer:
        :param parameters: List of `gpflow.Parameter` objects to send to TensorBoard if they are
        scalar. If None, all scalars will be sent to TensorBoard.
        :param additional_summaries: List of Summary objects to send to TensorBoard.
        """
        super().__init__(sequence, trigger)
        self.model = model
        all_summaries = [] if additional_summaries is None else additional_summaries
        parameters = model.parameters if parameters is None else parameters

        all_summaries += [tf.summary.scalar(p.full_name, p.constrained_tensor)
                          for p in parameters if p.size == 1]

        if not only_scalars:
            all_summaries += [tf.summary.histogram(p.full_name, p.constrained_tensor)
                              for p in parameters if p.size > 1]

        all_summaries.append(tf.summary.scalar("likelihood", model._likelihood_tensor))

        self.summary = tf.summary.merge(all_summaries)
        self.file_writer = file_writer
        self.global_step = global_step

    def run(self, ctx: ActionContext):
        step = ctx.iteration if self.global_step is None else ctx.session.run(self.global_step)
        summary = ctx.session.run(self.summary)
        self.file_writer.add_summary(summary, step)
