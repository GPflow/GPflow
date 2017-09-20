import abc
import tensorflow as tf

from gpflow.misc import GPflowError
from gpflow.base import Build
from gpflow.model import Model
from gpflow.training.external_optimizer import ScipyOptimizerInterface

class Optimizer:
    def __init__(self, model):
        if not isinstance(model, Model):
            raise ValueError('Incompatible type passed to optimizer: "{0}".'
                             .format(type(model)))
        self._model = model

    @abc.abstractmethod
    def minimize(self, *args, **kwargs):
        raise NotImplementedError('')

    def _pop_session(self, kwargs):
        session = kwargs.pop('session')
        if session is None:
            if self._model.session is None:
                raise ValueError('Session is not specified.')
            session = self._model.session
        return session

    def _pop_feed_dict(self, kwargs):
        return kwargs.pop('feed_dict', {})

    def _pop_maxiter(self, kwargs):
        return kwargs.pop('maxiter', 1000)

class ScipyOptimizer(Optimizer):
    def minimize(self, *args, **kwargs):
        pass

class TensorFlowOptimizer(Optimizer):
    def __init__(self, model, optimizer, *args, **kwargs):
        if isinstance(optimizer, type) and tf.train.Optimizer in optimizer.mro():
            self._optimizer = optimizer(*args, **kwargs)
        elif isinstance(optimizer, tf.train.Optimizer):
            self._optimizer = optimizer
        else:
            raise ValueError('Incorrect type of TensorFlow optimizer passed.')
        super(TensorFlowOptimizer, self).__init__(model)
        self._optimize_operation = None

    @property
    def optimize_operation(self):
        return self._optimize_operation

    def minimize(self, *args, **kwargs):
        session = self._pop_session(kwargs)
        if self._model.is_build_coherence(graph=session.graph) is Build.NO:
            raise GPflowError('Model is not built.')
        feed_dict = self._pop_feed_dict(kwargs)
        maxiter = self._pop_maxiter(kwargs)
        self._optimize_operation = self._optimizer.minimize(
            self._model.objective_tensor, *args, **kwargs)
        try:
            for _ in range(maxiter):
                session.run(self._optimize_operation, feed_dict=feed_dict)
        except KeyboardInterrupt:
            print("Optimization is interrupted.")
