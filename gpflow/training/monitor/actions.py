import enum
import glob
import os
import time
from typing import Callable, List, Iterator
from typing import Optional

import numpy as np
import tensorflow as tf

from ... import params_as_tensors_for
from ... import settings
from ...actions import Action, ActionContext, Watcher
from ...models import Model
from ...params import Parameter


# TODO: Make PrintAction print the timings, like before
# TODO: Change interface to have sequence and trigger do tasks every iteration by default
# TODO: Fix total iterations after a load of a model (need to use loaded global_step or something)
# TODO: Make sure that all monitor actions are run at the end of an optimisation run
# TODO: Implement something that does PrintAllTimings


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

    def __call__(self, context: Optional[ActionContext] = None, *, force_run: bool = False) -> None:
        """
        Action call method.
        The `action()` activates watcher, then calls `run` method and deactivates
        watcher even when `run` method raised an exception.

            :param context: Optional action context value. When argument is none
                the standard ActionContext is created and callee action assigns
                itself as a context owner.
        """
        context = ActionContext(self) if context is None else context
        if force_run or self._current_trigger_value(context) >= self._next:
            try:
                self.watcher.start()
                self.run(context)
            finally:
                self.watcher.stop()

            # Move to the next trigger time, and make sure it's after this current iteration
            if not force_run:
                while self._next <= self._current_trigger_value(context):
                    self._next = next(self._seq)


class PrintTimings(TriggeredAction):
    # Total iterations is currently broken after a load. See notebook after running it twice.
    # opt iterations is broken, since we don't have an "optimisation only" timer. Is there a way to do this, or remove?
    def __init__(self, sequence: Iterator, trigger: Trigger,
                 global_step: Optional[tf.Variable] = None, single_line: bool = True) -> None:
        super().__init__(sequence, trigger)
        self.global_step = global_step
        self.single_line = single_line

    def run(self, context: ActionContext) -> None:
        current_iter = context.iteration if context.iteration is not None else 0
        global_step_eval = context.iteration if self.global_step is None else context.session.run(self.global_step)
        if current_iter == 0:
            opt_iter = 0.0
            total_iter = 0.0
            last_iter = 0.0
        else:
            opt_iter = np.nan
            total_iter = global_step_eval / context.time_spent
            last_iter = (0.0 if not hasattr(self, '_last_iter')
                         else (global_step_eval - self._last_iter) / self._last_iter_timer.elapsed)

        print("\r%i, %i:\t%.2f optimisation iter/s\t%.2f total iter/s\t%.2f last iter/s" %
              (current_iter, global_step_eval, opt_iter, total_iter, last_iter), end='' if self.single_line else '\n')

        self._last_iter = global_step_eval
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
        """
        :param sequence:
        :param trigger:
        :param session:
        :param hist_path: Path to store checkpoint to.
        :param saver:
        :param restore_path: If None, will restore from `hist_path`. If False, it won't restore.
        :param global_step:
        """
        super().__init__(sequence, trigger)
        self.hist_path = hist_path
        self.restore_path = restore_path
        self.saver = tf.train.Saver(max_to_keep=3) if saver is None else saver
        self.session = session
        self.global_step = global_step

        if restore_path is None:
            if len(glob.glob(self.hist_path + "-*")) > 0:
                # History exists
                latest_step = np.max([int(os.path.splitext(os.path.basename(x))[0].split('-')[-1])
                                      for x in glob.glob(self.hist_path + "-*")])
                restore_path = self.hist_path + "-%i" % latest_step
        elif restore_path is False:
            restore_path = None

        if restore_path is not None:
            print("Restoring session from `%s`." % restore_path)
            self.saver.restore(session, restore_path)

    def run(self, ctx: ActionContext):
        self.saver.save(self.session, self.hist_path, global_step=self.global_step)


class ModelTensorBoard(TriggeredAction):
    def __init__(self, sequence: Iterator, trigger: Trigger, model: Model, file_writer: tf.summary.FileWriter, *,
                 only_scalars: bool = True,
                 parameters: Optional[List[Parameter]] = None,
                 additional_summaries: Optional[List[tf.Summary]] = None,
                 global_step: Optional[tf.Variable] = None) -> None:
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
        self.global_step = global_step
        all_summaries = [] if additional_summaries is None else additional_summaries
        parameters = model.parameters if parameters is None else parameters

        all_summaries += [tf.summary.scalar(p.full_name, tf.reshape(p.constrained_tensor, []))
                          for p in parameters if p.size == 1]

        if not only_scalars:
            all_summaries += [tf.summary.histogram(p.full_name, p.constrained_tensor)
                              for p in parameters if p.size > 1]

        all_summaries.append(tf.summary.scalar("likelihood", model._likelihood_tensor))

        self.summary = tf.summary.merge(all_summaries)
        self.file_writer = file_writer

    def run(self, ctx: ActionContext):
        summary = ctx.session.run(self.summary)
        step = ctx.session.run(self.global_step) if self.global_step is not None else ctx.iteration
        self.file_writer.add_summary(summary, step)


class LmlTensorBoard(ModelTensorBoard):
    """
    Only outputs a full LML of the model to TensorBoard, computed in minibatches.
    """

    def __init__(self, sequence: Iterator, trigger: Trigger, model: Model, file_writer: tf.summary.FileWriter, *,
                 minibatch_size: Optional[int] = 100, global_step: Optional[tf.Variable] = None,
                 verbose: Optional[bool] = True):
        super().__init__(sequence, trigger, model, file_writer, global_step=global_step)
        self.minibatch_size = minibatch_size
        self._full_lml = tf.placeholder(settings.tf_float, shape=())
        self.summary = tf.summary.scalar("full_lml", self._full_lml)
        self.verbose = verbose

    def run(self, ctx: ActionContext):
        with params_as_tensors_for(self.model):
            tfX, tfY = self.model.X, self.model.Y

        if self.verbose:  # pragma: no cover
            try:
                import tqdm
                wrapper = tqdm.tqdm
            except ModuleNotFoundError:
                print("monitor: for `verbose=True`, install `tqdm`.")
                wrapper = lambda x: x
            print("")
        else:  # pragma: no cover
            wrapper = lambda x: x

        lml = 0.0
        num_batches = -(-len(self.model.X._value) // self.minibatch_size)  # round up
        for mb in wrapper(range(num_batches)):
            start = mb * self.minibatch_size
            finish = (mb + 1) * self.minibatch_size
            Xmb = self.model.X._value[start:finish, :]
            Ymb = self.model.Y._value[start:finish, :]
            mb_lml = self.model.compute_log_likelihood(feed_dict={tfX: Xmb, tfY: Ymb})
            lml += mb_lml * len(Xmb)
        lml = lml / len(self.model.X._value)

        summary, step = ctx.session.run([self.summary, self.global_step], feed_dict={self._full_lml: lml})
        print("Full lml: %f (%.2e)" % (lml, lml))
        self.file_writer.add_summary(summary, step)


def seq_exp_lin(growth, max, start=1.0, start_jump=None):
    """
    Returns an iterator that constructs a sequence beginning with `start`, growing exponentially:
    the step size starts out as `start_jump` (if given, otherwise `start`), multiplied by `growth`
    in each step. Once `max` is reached, growth will be linear with `max` step size.
    """
    start_jump = start if start_jump is None else start_jump
    gap = start_jump
    last = start - start_jump
    while 1:
        yield gap + last
        last = last + gap
        gap = min(gap * growth, max)
