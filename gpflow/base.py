from typing import Optional, Union, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from .util import default_float


DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]
Transform = tfp.bijectors.Bijector
Prior = tfp.distributions.Distribution


positive = tfp.bijectors.Softplus
triangular = tfp.bijectors.FillTriangular


class Parameter(tf.Variable):
    def __init__(self,
                 value, *,
                 transform: Optional[Transform] = None,
                 prior: Optional[Prior] = None,
                 trainable: bool = True,
                 dtype: DType = None,
                 name: str =None):
        """
        Unconstrained parameter representation.
        According to standart terminology `y` is always transformed representation or,
        in other words, it is constrained version of the parameter. Normally, it is hard
        to operate with unconstrained parameters. For e.g. `variance` cannot be negative,
        therefore we need positive constraint and it is natural to use constrained values.
        """
        if isinstance(value, tf.Variable):
            self._unconstrained = value
        else:
            value = _to_unconstrained(value, transform)
            self._unconstrained = tf.Variable(value, dtype=dtype, name=name, trainable=trainable)

        self.prior = prior
        self._transform = transform

    def log_prior(self):
        x = self.read_value()
        y = self._unconstrained
        dtype = x.dtype

        log_prob = tf.convert_to_tensor(0., dtype=dtype)
        log_det_jacobian = tf.convert_to_tensor(0., dtype=dtype)

        bijector = self.transform
        if self.prior is not None:
            log_prob = self.prior.log_prob(x)
        if self.transform is not None:
            log_det_jacobian = bijector.forward_log_det_jacobian(y, y.shape.ndims)
        return log_prob + log_det_jacobian

    @property
    def handle(self):
        return self._unconstrained.handle

    def value(self):
        return _to_constrained(self._unconstrained.value(), self.transform)

    def read_value(self):
        return _to_constrained(self._unconstrained.read_value(), self.transform)

    @property
    def transform(self):
        return self._transform

    @property
    def trainable(self):
        return self._unconstrained.trainable

    @trainable.setter
    def trainable(self, flag: Union[bool, int]):
        self._unconstrained._trainable = bool(flag)

    @property
    def initial_value(self):
        return self._unconstrained.initial_value

    def assign(self, value, use_locking=False, name=None, read_value=True):
        unconstrained_value = _to_unconstrained_data(value, self.transform)
        self._unconstrained.assign(unconstrained_value,
            read_value=read_value,
            use_locking=use_locking)

    @property
    def name(self):
        return self._unconstrained.name

    @property
    def initializer(self):
        return self._unconstrained.initializer

    @property
    def device(self):
        return self._unconstrained.device

    @property
    def dtype(self):
        return self._unconstrained.dtype

    @property
    def graph(self):
        return self._unconstrained.graph

    @property
    def op(self):
        return self._unconstrained.op

    @property
    def shape(self):
        return self._unconstrained.shape

    def get_shape(self):
        return self.shape

    def _should_act_as_resource_variable(self):
        pass

    def __repr__(self):
        return self.read_value().__repr__()

    def __ilshift__(self, value: VariableData) -> 'Parameter':
        self.assign(value)
        return self


Parameter._OverloadAllOperators()
tf.register_tensor_conversion_function(Parameter, lambda x, *args, **kwds: x.read_value())


def _to_unconstrained_data(data: VariableData, dtype: Optional[DType] = None) -> np.ndarray:
    if isinstance(data, (tf.Tensor, tf.Variable)):
        if dtype is not None and data.dtype != dtype:
            return tf.cast(data, dtype)
        return data
    if dtype is not None and isinstance(dtype, tf.DType):
        dtype = dtype.as_numpy_dtype
    if dtype is None and not isinstance(data, np.ndarray):
        dtype = default_float()
    return np.array(data, dtype=dtype)


def _to_constrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.forward(value)
    return value


def _to_unconstrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.inverse(value)
    return value
