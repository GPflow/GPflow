import enum
import time
from typing import Optional

import numpy as np

from ...actions import Action, ActionContext


# TODO: Make TriggeredAction, which allows a sequence to be passed of iterations or times for the action to be run.
# TODO: Implement TensorBoard action
# TODO: Implement model saving action


@enum.unique
class Trigger(enum.Enum):
    TOTAL_TIME = 1
    OPTIMISATION_TIME = 2
    ITER = 3


class PrintAction(Action):
    def __init__(self, model, text, *, single_line=False):
        self.model = model
        self.text = text
        self.single_line = single_line

    def run(self, ctx: ActionContext):
        likelihood = ctx.session.run(self.model.likelihood_tensor)
        print('{}: iteration {} likelihood {:.4f}'.format(self.text, ctx.iteration, likelihood),
              end="\r" if self.single_line else "\n")


class SleepAction(Action):
    def __init__(self, sleep_seconds):
        self.sleep_seconds = sleep_seconds

    def run(self, ctx):
        time.sleep(self.sleep_seconds)


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
        while self._next <= self._current_trigger_value(context):
            self._next = next(self._seq)
