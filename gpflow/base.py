from typing import Optional, Union, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]
Transform = tfp.bijectors.Bijector
Prior = tfp.distributions.Distribution


class Parameter(tfe.Variable):
    """
    Unconstrained parameter representation.
    According to standart terminology `y` is always transformed representation or,
    in other words, it is constrained version of the parameter. Normally, it is hard
    to operate with unconstrained parameters. For e.g. `variance` cannot be negative,
    therefore we need positive constraint and it is natural to use constrained values.
    """
    def __init__(self, data: VariableData,
                 transform: Optional[Transform] = None,
                 prior: Optional[Prior] = None,
                 trainable=True,
                 dtype=None):
        data = _to_variable_data(data, dtype=dtype)
        unconstrained_data = _to_unconstrained(data, transform)
        super().__init__(unconstrained_data, trainable=trainable)
        self.transform = transform
        self.prior = prior

    @property
    def trainable(self) -> bool:
        return super().trainable

    @trainable.setter
    def trainable(self, flag: Union[bool, int]):
        self._trainable = bool(flag)

    @property
    def constrained(self):
        if self.transform is None:
            return self
        return self.transform.forward(self)

    def log_prior(self):
        x = self.constrained
        y = self
        log_prob = 0.
        bijector = self.transform
        if self.prior is not None:
            log_prob = self.prior.log_prob(x)
        log_det_jacobian = 0.
        if self.transform is not None:
            log_det_jacobian = bijector.forward_log_det_jacobian(y, (0))
        return log_prob + log_det_jacobian

    def __ilshift__(self, data: VariableData):
        data = _to_variable_data(data)
        unconstrained_data = _to_unconstrained(data, self.transform)
        self._init_from_args(unconstrained_data, trainable=self.trainable)
        return self


class Module:
    def __init__(self):
        self._modules = dict()
        self._parameters = dict()

    @property
    def variables(self) -> List[tfe.Variable]:
        self_variables = list(self._parameters.values())
        modules_variables = sum([m.variables for m in self._modules.values()], [])
        return self_variables + modules_variables

    @property
    def trainable_variables(self) -> List[tfe.Variable]:
        return [v for v in self.variables if v.trainable]

    def __getattr__(self, name):
        if name in self._parameters:
            parameter = self._parameters[name]
            return parameter.constrained
        elif name in self._modules:
            return self._modules[name]

    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        super().__setattr__(name, value)


def _to_variable_data(data: VariableData, dtype: Optional[DType] = None) -> np.ndarray:
    if isinstance(data, (tf.Tensor, tfe.Variable)):
        if dtype is not None and data.dtype != dtype:
            return tf.cast(data, dtype)
        return data
    if dtype is not None and isinstance(dtype, tf.DType):
        dtype = dtype.as_numpy_dtype
    return np.array(data, dtype=dtype)


def _to_unconstrained(data: VariableData, transform: Transform) -> tf.Tensor:
    unconstrained_data = data
    if transform is not None:
        unconstrained_data = transform.inverse(data)
    return unconstrained_data


class ModuleList:
    pass
