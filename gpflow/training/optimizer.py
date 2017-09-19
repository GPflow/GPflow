import abc

from gpflow.model import Model
from gpflow.training.external_optimizer import ScipyOptimizerInterface

class Optimizer:
    def __init__(self, model, var_list=None):
        if not isinstance(model, Model):
            raise ValueError('Incompatible type passed to optimizer: "{0}".'
                             .format(type(model)))
        self._var_list = var_list
        self._model = model

    @abc.abstractmethod
    def minimize(self, *args, **kwargs):
        raise NotImplementedError('')

class ScipyOptimizer(Optimizer):
    def minimize(self, *args, **kwargs):
        pass

class TensorFlowOptimizer(Optimizer):
    def minimize(self, *args, **kwargs):
        pass
