# Copyright 2018 Artem Artemev @awav
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

import abc
import collections
import inspect
import itertools
from timeit import default_timer as timer
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np
import tensorflow as tf

from . import get_default_session
from .models import Model


class Watcher:
    """
    Watcher is a sort of time manager.
    
    It holds information when timer is started, and when it is stopped.
    If elapsed time is asked and timer is not stopped, then different between
    current time and started is returned. Otherwise, different between
    start and stop time.
    """

    def __init__(self):
        self._start = None
        self._stop = None

    def start(self):
        """
        Set start time [µs].
        """
        self._start = timer()
    
    def stop(self):
        """
        Set stop time [µs].
        """
        self._stop = timer()
    
    @property
    def elapsed(self):
        """
        Elapsed time [µs] between start and stop timestamps. If stop is empty then
        returned time is difference between start and current timestamp.
        """
        if self._stop is None:
            return timer() - self._start
        return self._stop - self._start
    

class ActionContext:
    """
    Action context contains sharable data between sequence of action runs.

        :param owner: Action to which context originally belongs to.
        :param session: Tensorflow session which will be used when necessary by
            all dependent actions ran by owner.
    """

    def __init__(self, owner: 'Action', session: Optional[tf.Session] = None):
        if session is None:
            session = tf.get_default_session()
            session = get_default_session() if session is None else session
        self.session = session
        self.owner = owner
    
    @property
    def iteration(self) -> int:
        """
        Current iteration number.
        In fact, this is proxy method for Loop's owner interaction.

            :return: Iteration number or None, in case when owner doesn't
                have any references to iteration.
        """
        return _get_attr(self.owner, iteration=None)
    
    @property
    def time_spent(self) -> float:
        """
        Elapsed time since owner started running dependent actions.

            :return: Spent time [µs] by owner on running up to the call.
        """
        return self.owner.watcher.elapsed


class Action(metaclass=abc.ABCMeta):
    """
    Action base abstraction for wrapping functionality inside function-like container.
    
    Actions are functions with extra utilities. Each action has watcher
    which gauge how much time action spent on execution. Action may contain
    information about iteration, but usually 
    """
    @abc.abstractmethod
    def run(self, context: ActionContext) -> None:
        """
        Run action's execution. Must be implemented by decendants.

            :param context: Required action context.
        """
        pass
    
    @property
    def watcher(self) -> Watcher:
        """
        Gives an access to action's watcher.

            :return: Action's watcher instance.
        """
        if not hasattr(self, "_watcher"):
            self._watcher = Watcher()
        return self._watcher
    
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
        try:
            self.watcher.start()
            self.run(context)
        finally:
            self.watcher.stop()


# ======================
# Action implementations
# ======================

class Loop(Action):
    """
    Action loop abstraction.
    Default loop is an infinite with starting point at zero and update step one. 
    It evaluates body action as many times as it is specified by start, stop and step.
    Loop provides two control signals - 'Break' and 'Continue'.
    These signals must be raised as a standard exception. Depending on which signal
    was raised the loop either stops execution or continues from beginning.

        :param action: Action or list of actions, which will be run within loop.
        :param stop: Maximum loop iteration.
        :param start: Starting point for loop iterations.
        :param step: Integer interval between iterations.
    """

    class Continue(Exception):
        """Continue signal returns execution to the beginning of the loop."""
        pass

    class Break(Exception):
        """Break signal stops loop immediately."""
        pass

    def __init__(self,
            action: Action,
            stop: Optional[int] = None,
            start: int = 0,
            step: int = 1) -> None:
        self._action = _try_convert_action(action)
        self.start = start
        self.stop = stop
        self.step = step
    
    @property
    def iteration(self) -> int:
        """Active iteration number."""
        return _get_attr(self, _iteration=None)
    
    def with_iteration(self, iteration: int):
        """
        Set iteration number.
        
            :param iteration: Iteration interger number.
            :return: Loop itself.
        """
        self._iteration = iteration
        return self

    def with_action(self, action: Action) -> 'Loop':
        """
        Set loop body action.
            
            :param action: Loop body action.
            :return: Loop itself.
        """
        self._action = _try_convert_action(action)
        return self

    def with_settings(self,
            stop: Optional[int] = None,
            start: int = 0,
            step: int = 1) -> 'Loop':
        """
        Set start, stop and step loop configuration.

            :param stop: Looop stop iteration integer. If None then loop
                becomes infinite.
            :param start: Loop iteration start integer.
            :param step: Loop iteration interval integer.
            :return: Loop itself.
        """
        self.start = start
        self.stop = stop
        self.step = step
        return self

    def run(self, context: ActionContext):
        """
        Run performs loop iterations.

            :param context: Action context.
        """
        iterator = itertools.count(start=self.start, step=self.step)
        for i in iterator:
            self.with_iteration(i)
            if self.stop is not None and i >= self.stop:
                break
            try:
                self._action(context)
            except Loop.Continue:
                continue
            except Loop.Break:
                break
    

class Group(Action):
    """
    Group is an ordered list of actions.
    Group runs list of actions step by step, executing them separately
    in single for loop.

        :param actions: List of actions.
    """
    def __init__(self, *actions: Sequence[Action]) -> None:
        self._actions = actions
    
    def run(self, context: ActionContext):
        """
        Run executes list of cached actions in the for loop.
        """
        for action in self._actions:
            action(context)


class Condition(Action):
    """
    Condition actions is a branching mechanism.
    Depending on boolean output of callable conditional function, either
    true or false passed action is executed.

        :param condition_fn: Boolean function with single input, current context.
        :param action_true: True action, which is executed when output of
            conditional function returns true.
        :param action_false: False action, which is executed when output of
            conditional function returns false.
    """
    def __init__(self,
            condition_fn: Callable[[ActionContext], bool],
            action_true: Action,
            action_false: Optional[Action] = None) -> None:
        self.condition_fn = condition_fn
        self.action_false = action_false
        self.action_true = action_true

    def run(self, context: ActionContext):
        cond = self.condition_fn(context)
        if cond:
            return self.action_true(context)
        elif self.action_false is None:
            return None
        return self.action_false(context)


class Optimization(Action):
    """
    Optimization is an action wrapper for GPflow optimizers.
    It contains an optimizer, a model and TensorFlow minimization tensor
    generated by the optimizer and model.
    This action may be used as single optimization step.
    """

    def with_optimizer(self, optimizer) -> 'Optimization':
        """
        Replace optimizer.

            :param optimizer: GPflow optimizer.
            :return: Optimization instance self reference.
        """
        self._optimizer = optimizer
        return self


    def with_model(self, model: Model) -> 'Optimization':
        """
        Replace model.

            :param model: GPflow model.
            :return: Optimization instance self reference.
        """
        self._model = model
        return self
    
    def with_optimizer_tensor(self, tensor: Union[tf.Tensor, tf.Operation]) -> 'Optimization':
        """
        Replace optimizer tensor.

            :param model: Tensorflow tensor.
            :return: Optimization instance self reference.
        """
        self._optimizer_tensor = tensor
        return self
    
    def with_run_kwargs(self, **kwargs: Dict[str, Any]) -> 'Optimization':
        """
        Replace Tensorflow session run kwargs.
        Check Tensorflow session run [documentation](https://www.tensorflow.org/api_docs/python/tf/Session).

            :param kwargs: Dictionary of tensors as keys and numpy arrays or
                primitive python types as values.
            :return: Optimization instance self reference.
        """
        self._run_kwargs = kwargs
        return self

    @property
    def model(self) -> Model:
        """The `model` is an attribute for getting optimization's model."""
        return _get_attr(self, _model=None)

    @property
    def optimizer_tensor(self) -> Union[tf.Tensor, tf.Operation]:
        """
        The `optimizer_tensor` is an attribute for getting optimization's
        optimizer tensor.
        """
        return _get_attr(self, _optimizer_tensor=None)
    
    @property
    def run_kwargs(self):
        """The `run_kwargs` is an attribute for getting session run's kwargs."""
        return _get_attr(self, _run_kwargs=None)
    
    def run(self, context: ActionContext) -> None:
        context.session.run(self.optimizer_tensor, **self.run_kwargs)


def _get_attr(obj, **attr):
    assert len(attr) == 1
    return getattr(obj, list(attr.keys())[0], list(attr.values())[0])


def _is_action(a):
    return isinstance(a, Action) or (callable(a) and len(inspect.signature(a).parameters) == 1)


def _try_convert_action(action):
    if _is_action(action):
        return action
    if not isinstance(action, list):
        raise ValueError('Expected either action type of list of actions.')
    for a in action:
        if _is_action(a):
            continue
        raise ValueError('List consists of non homogeneous action types.')
    return Group(*action)