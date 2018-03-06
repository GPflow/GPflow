import abc

class ActionContext:
    pass


class Action(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __call__(self, context) -> 'ActionContext':
        pass


class Loop(Action):
    def __init__(self, action, start=0, stop=None, step=1) -> None:
        self._action = action
        self._start = start
        self._stop = stop
        self._step = step

    def __call__(self, context) -> 'ActionContext':
        for iteration in range(self._start, self._stop, self._step):
            context = self._action(context)
        return context


class Sequence(Action):
    def __init__(self, *actions) -> None:
        self._actions = actions
    
    def __call__(self, context):
        for action in self._actions:
            context = action(context)
        return context


class Condition(Action):
    def __init__(self, condition_fn, action_true, action_false) -> None:
        self._condition_fn = condition_fn
        self._action_true = action_true
        self._action_false = action_false

    def __call__(self, context) -> 'ActionContext':
        if self._condition_fn(context):
            return self._action_true(context)
        return self._action_false(context)


class Optimization(Action):
    pass


class Tensorboard(Action):
    pass


class Logger(Action):
    pass