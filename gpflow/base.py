from typing import Optional, Union, List, Tuple
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp
from .util import default_float


DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]
Transform = tfp.bijectors.Bijector
Prior = tfp.distributions.Distribution
ModuleLike = Union['Module', 'ModuleList']


positive = tfp.bijectors.Softplus
triangular = tfp.bijectors.FillTriangular


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
                 trainable: bool = True,
                 dtype: np.dtype = None,
                 unconstrained: bool = False,
                 constrained: bool = True):
        data = _to_variable_data(data, dtype=dtype)
        unconstrained_data = _to_unconstrained(data, transform)
        super().__init__(unconstrained_data, trainable=trainable)
        self.transform = transform
        self.prior = prior
        self.is_constrained = constrained

    @property
    def trainable(self) -> bool:
        return super().trainable

    # TODO(@awav): proper solution
    # @property
    # def shape(self):
    #     return self._shape

    #     # if self.is_constrained:
    #     #     return self._read_variable_op().shape
    #     # return super().shape

    # @property
    # def is_constrained(self):
    #     constrained = self.__dict__.get('_is_constrained')
    #     return bool(constrained)

    # @is_constrained.setter
    # def is_constrained(self, value: bool):
    #     value = bool(value)
    #     constrained = self.__dict__.get('_is_constrained')
    #     if constrained is not None and value == constrained:
    #         return
    #     if value:
    #         shape = self.constrained.shape
    #     else:
    #         shape = super()._read_variable_op().shape
    #     self._shape = shape
    #     self._is_constrained = value
    #
    # def _read_variable_op(self):
    #     value = super()._read_variable_op()
    #     constrained = self.__dict__.get('_is_constrained')
    #     if self.transform is None or (constrained is not None and not constrained):
    #         return value
    #     return self.transform.forward(value)
    #
    # @property
    # def unconstrained(self):
    #     if self.transform is None:
    #         return self
    #     return self.transform.inverse(self)

    @property
    def constrained(self):
        if self.transform is None:
            return self
        return self.transform.forward(self)

    @trainable.setter
    def trainable(self, flag: Union[bool, int]):
        self._trainable = bool(flag)

    def log_prior(self):
        x = self.constrained
        y = self
        dtype = x.dtype
        log_prob = tf.convert_to_tensor(0., dtype=dtype)
        bijector = self.transform
        prior_exists = self.prior is not None
        transform_exists = self.transform is not None
        if prior_exists:
            log_prob = self.prior.log_prob(x)
        log_det_jacobian = tf.convert_to_tensor(0., dtype=dtype)
        if transform_exists:
            log_det_jacobian = bijector.forward_log_det_jacobian(y, y.shape.ndims)
        return log_prob + log_det_jacobian

    def __call__(self):
        return self.constrained

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

    @property
    def trainable(self):
        return self.trainable_variables != []

    @trainable.setter
    def trainable(self, value: bool):
        for variable in self.variables:
            variable.trainable = value

    def __getattr__(self, name):
        if name in self.__dict__['_parameters']:
            return self.__dict__['_parameters'].get(name)
        elif name in self._modules:
            return self._modules[name]
        raise AttributeError

    def __setattr__(self, name, value):
        parameters = self.__dict__.get('_parameters')
        is_parameter = isinstance(value, (Parameter, tfe.Variable))
        if is_parameter and parameters is None:
            raise AttributeError()
        if parameters is not None and name in parameters and not is_parameter:
            raise AttributeError()
        if is_parameter and parameters is not None:
            parameters[name] = value
            return

        modules = self.__dict__.get('_modules')
        is_module = isinstance(value, (Module, ModuleList))
        if is_module and modules is None:
            raise AttributeError()
        if modules is not None and name in modules and not is_module:
            raise AttributeError()
        if is_module and modules is not None:
            self._modules[name] = value
            return

        super().__setattr__(name, value)


class ModuleList:
    def __init__(self, modules: List[ModuleLike]):
        self._modules = modules

    @property
    def variables(self) -> List[tfe.Variable]:
        module_variables = sum([m.variables for m in self._modules], [])
        return module_variables

    @property
    def trainable_variables(self) -> List[tfe.Variable]:
        return [v for v in self.variables if v.trainable]

    def __len__(self):
        return len(self._modules)

    def append(self, module: ModuleLike):
        self._modules.append(module)

    def __getitem__(self, index: int) -> ModuleLike:
        return self._modules[index]

    def __setitem__(self, index: int, module: ModuleLike):
        self._modules[index] = module


def _to_variable_data(data: VariableData, dtype: Optional[DType] = None) -> np.ndarray:
    if isinstance(data, (tf.Tensor, tfe.Variable)):
        if dtype is not None and data.dtype != dtype:
            return tf.cast(data, dtype)
        return data
    if dtype is not None and isinstance(dtype, tf.DType):
        dtype = dtype.as_numpy_dtype
    if dtype is None and not isinstance(data, np.ndarray):
        dtype = default_float()
    return np.array(data, dtype=dtype)


def _to_unconstrained(data: VariableData, transform: Transform) -> tf.Tensor:
    unconstrained_data = data
    if transform is not None:
        unconstrained_data = transform.inverse(data)
    return unconstrained_data
