import functools
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import array_ops

from .config import default_float

DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]
TensorLike = object  # Union[tf.Tensor, tf.Variable, np.ndarray], but doesn't work with multipledispatch
Transform = tfp.bijectors.Bijector
Prior = tfp.distributions.Distribution


def _IS_PARAMETER(o):
    return isinstance(o, Parameter)


def _IS_TRAINABLE_PARAMETER(o):
    return (_IS_PARAMETER(o) and o.trainable)


class Module(tf.Module):
    @property
    def parameters(self):
        return tuple(self._flatten(predicate=_IS_PARAMETER))

    @property
    def trainable_parameters(self):
        return tuple(self._flatten(predicate=_IS_TRAINABLE_PARAMETER))


class Parameter(tf.Module):
    def __init__(self,
                 value,
                 *,
                 transform: Optional[Transform] = None,
                 prior: Optional[Prior] = None,
                 prior_on_constrained: bool = True,
                 trainable: bool = True,
                 dtype: Optional[DType] = None,
                 name: Optional[str] = None):
        """
        A parameter retains both constrained and unconstrained
        representations, If no transforms is provided, these two values will be the same.
        It is often challenging to operate with unconstrained parameters. For example a variance cannot be negative,
        therefore we need a positive constraint and it is natural to use constrained values.
        A prior can be imposed either on the constrained or unconstrained version of the parameter.
        """
        super().__init__()

        value = _verified_value(value, dtype)
        if isinstance(value, tf.Variable):
            self._unconstrained = value
        else:
            value = _to_unconstrained(value, transform)
            self._unconstrained = tf.Variable(value, dtype=dtype, name=name, trainable=trainable)

        self.prior = prior
        self.prior_on_constrained = prior_on_constrained
        self._transform = transform

    def log_prior(self, evaluate_on_constrained: bool = True):
        """ Prior probability density.
        This can be evaluated either on the constrained or unconstrained variable.
        For example if transform = Exp(), prior = Uniform(), then log_prior will either be
        uniform (when evaluate_on_constrained is True), or scale as 1/value when
        evaluate_on_constrained is set to False.
        """

        x = self._unconstrained
        y = self.read_value()

        if self.prior is not None:
            z = y if self.prior_on_constrained else x
            out = tf.reduce_sum(self.prior.log_prob(z))

            if self.transform is not None:
                # If the requested prior probability density does not match the
                # variable on which the prior is defined, we need to make use of the
                # Jacobian to compensate.
                if evaluate_on_constrained and not self.prior_on_constrained:
                    log_det_jacobian = self.transform.inverse_log_det_jacobian(y, y.shape.ndims)
                    out += tf.reduce_sum(log_det_jacobian)
                if not evaluate_on_constrained and self.prior_on_constrained:
                    # This is the original definition used by gpflow
                    log_det_jacobian = self.transform.forward_log_det_jacobian(x, x.shape.ndims)
                    out += tf.reduce_sum(log_det_jacobian)
            return out
        else:
            return tf.convert_to_tensor(0., dtype=self.dtype)

    def value(self):
        return _to_constrained(self._unconstrained.value(), self.transform)

    def read_value(self):
        return _to_constrained(self._unconstrained.read_value(), self.transform)

    def experimental_ref(self):
        return self

    def deref(self):
        return self

    @property
    def unconstrained_variable(self):
        return self._unconstrained

    @property
    def transform(self):
        return self._transform

    @transform.setter
    def transform(self, new_transform):
        constrained_value = self.read_value()
        self._transform = new_transform
        self.assign(constrained_value)

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
        # TODO(sergio.pasc): Find proper solution for casting / Discuss solution
        value = _verified_value(value, self.dtype)
        unconstrained_value = _to_unconstrained(value, self.transform)

        self._unconstrained.assign(unconstrained_value, read_value=read_value, use_locking=use_locking)

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
    def op(self):
        return self._unconstrained.op

    @property
    def shape(self):
        if self.transform is not None:
            return self.transform.forward_event_shape(self._unconstrained.shape)
        return self._unconstrained.shape

    def numpy(self):
        return self.read_value().numpy()

    def get_shape(self):
        return self.shape

    def _should_act_as_resource_variable(self):
        pass

    @property
    def handle(self):
        return self._unconstrained.handle

    def __repr__(self):
        return self.read_value().__repr__()

    # Below
    # TensorFlow copy-paste code to make variable-like object to work

    @classmethod
    def _OverloadAllOperators(cls):  # pylint: disable=invalid-name
        """Register overloads for all operators."""
        for operator in tf.Tensor.OVERLOADABLE_OPERATORS:
            cls._OverloadOperator(operator)
        # For slicing, bind getitem differently than a tensor (use SliceHelperVar
        # instead)
        # pylint: disable=protected-access
        setattr(cls, "__getitem__", array_ops._SliceHelperVar)

    @classmethod
    def _OverloadOperator(cls, operator):  # pylint: disable=invalid-name
        """Defer an operator overload to `ops.Tensor`.

        We pull the operator out of ops.Tensor dynamically to avoid ordering issues.

        Args:
            operator: string. The operator name.
        """
        tensor_oper = getattr(tf.Tensor, operator)

        def _run_op(a, *args, **kwargs):
            # pylint: disable=protected-access
            return tensor_oper(a.read_value(), *args, **kwargs)

        functools.update_wrapper(_run_op, tensor_oper)
        setattr(cls, operator, _run_op)

    # NOTE(mrry): This enables the Variable's overloaded "right" binary
    # operators to run when the left operand is an ndarray, because it
    # accords the Variable class higher priority than an ndarray, or a
    # numpy matrix.
    # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
    # mechanism, which allows more control over how Variables interact
    # with ndarrays.
    __array_priority__ = 100


Parameter._OverloadAllOperators()
tf.register_tensor_conversion_function(Parameter, lambda x, *args, **kwds: x.read_value())


def _verified_value(value: VariableData, dtype: Optional[DType] = None) -> np.ndarray:
    if isinstance(value, tf.Variable):
        return value
    if dtype is None:
        dtype = default_float()
    return tf.cast(value, dtype)


def _to_constrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.forward(value)
    return value


def _to_unconstrained(value: VariableData, transform: Transform) -> tf.Tensor:
    if transform is not None:
        return transform.inverse(value)
    return value
