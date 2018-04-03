import enum
import glob
import os
import time
from typing import Optional

import numpy as np
import tensorflow as tf

from ...actions import Action, ActionContext, Watcher


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
    def __init__(self, sequence, trigger: Trigger):
        super().__init__()
        self._seq = sequence
        self.trigger = trigger
        self._next = next(self._seq) if self._seq is not None else np.inf

    def _current_trigger_value(self, context):
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


class PrintAction(TriggeredAction):
    def __init__(self, sequence, trigger, model, text, *, single_line=False):
        super().__init__(sequence, trigger)
        self.model = model
        self.text = text
        self.single_line = single_line

    def run(self, ctx: ActionContext):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood),
              end="\r" if self.single_line else "\n")


class CallbackAction(TriggeredAction):
    def __init__(self, sequence, trigger, callback):
        super().__init__(sequence, trigger)
        self._callback = callback

    def run(self, ctx: ActionContext):
        self._callback()


class SleepAction(TriggeredAction):
    def __init__(self, sequence, trigger, sleep_seconds):
        super().__init__(sequence, trigger)
        self.sleep_seconds = sleep_seconds

    def run(self, ctx):
        time.sleep(self.sleep_seconds)


class StoreSession(TriggeredAction):
    def __init__(self, sequence, trigger: Trigger, session: tf.Session, hist_path, saver=None, restore_path=None,
                 global_step=None):
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

    def run(self, ctx):
        self.saver.save(self.session, self.hist_path, global_step=self.global_step)


class PrintTimings(TriggeredAction):
    def __init__(self, sequence, trigger, global_step=None, single_line=True):
        super().__init__(sequence, trigger)
        self.global_step = global_step
        self.single_line = single_line

    def run(self, context):
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
