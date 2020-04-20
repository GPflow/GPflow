import functools
from enum import Enum
from typing import Any, List, Optional, Sequence, TYPE_CHECKING, Tuple, Union

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.python.ops import array_ops
from typing_extensions import Final

from .config import default_float, default_summary_fmt

if TYPE_CHECKING:
    from IPython.lib import pretty

DType = Union[np.dtype, tf.DType]
VariableData = Union[List, Tuple, np.ndarray, int, float]  # deprecated
Transform = Union[tfp.bijectors.Bijector]
Prior = Union[tfp.distributions.Distribution]


TensorType = Union[tf.Tensor, tf.Variable, "Parameter"]
"""
Type alias for tensor-like types that are supported by most TensorFlow and GPflow operations.

NOTE: Union types like this do not work with the `register` method of `multipledispatch`'s
`Dispatcher` class. Instead use `TensorLike`.
"""


# We've left this as object until we've tested the performance consequences of using the full set
# (np.ndarray, tf.Tensor, tf.Variable, Parameter), see https://github.com/GPflow/GPflow/issues/1434
TensorLike: Final[Tuple[type, ...]] = (object,)
"""
:var TensorLike: Collection of tensor-like types for registering implementations with
    `multipledispatch` dispatchers.
"""


_NativeScalar = Union[int, float]
_Array = Sequence[Any]  # a nested array of int, float, bool etc. kept simple for readability
TensorData = Union[_NativeScalar, _Array, TensorType]


def _IS_PARAMETER(o: Any) -> bool:
    return isinstance(o, Parameter)


def _IS_TRAINABLE_PARAMETER(o: Any) -> bool:
    return _IS_PARAMETER(o) and o.trainable


class Module(tf.Module):
    @property
    def parameters(self) -> Tuple["Parameter", ...]:
        return tuple(self._flatten(predicate=_IS_PARAMETER))

    @property
    def trainable_parameters(self) -> Tuple["Parameter", ...]:
        return tuple(self._flatten(predicate=_IS_TRAINABLE_PARAMETER))

    def _representation_table(self, object_name: str, tablefmt: Optional[str]) -> str:
        from .utilities import leaf_components, tabulate_module_summary

        repr_components = [object_name]
        if leaf_components(self):
            repr_components.append(tabulate_module_summary(self, tablefmt=tablefmt))
        return "\n".join(repr_components)

    def _repr_html_(self) -> str:
        """ Nice representation of GPflow objects in IPython/Jupyter notebooks """
        from html import escape

        return self._representation_table(escape(repr(self)), "html")

    def _repr_pretty_(self, p: "pretty.RepresentationPrinter", cycle: bool) -> None:
        """ Nice representation of GPflow objects in the IPython shell """
        repr_str = self._representation_table(repr(self), default_summary_fmt())
        p.text(repr_str)


class PriorOn(Enum):
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"


class Parameter(tf.Module):
    def __init__(
        self,
        value: TensorData,
        *,
        transform: Optional[Transform] = None,
        prior: Optional[Prior] = None,
        prior_on: Union[str, PriorOn] = PriorOn.CONSTRAINED,
        trainable: bool = True,
        dtype: Optional[DType] = None,
        name: Optional[str] = None,
    ):
        """
        A parameter retains both constrained and unconstrained
        representations. If no transform is provided, these two values will be the same.
        It is often challenging to operate with unconstrained parameters. For example, a variance cannot be negative,
        therefore we need a positive constraint and it is natural to use constrained values.
        A prior can be imposed either on the constrained version (default) or on the unconstrained version of the parameter.
        """
        super().__init__()

        self._transform = transform
        self.prior = prior
        self.prior_on = prior_on

        if isinstance(value, tf.Variable):
            self._unconstrained = value
        else:
            unconstrained_value = self.validate_unconstrained_value(value, dtype)
            self._unconstrained = tf.Variable(
                unconstrained_value, dtype=dtype, name=name, trainable=trainable
            )

    def log_prior_density(self) -> tf.Tensor:
        """ Log of the prior probability density of the constrained variable. """

        if self.prior is None:
            return tf.convert_to_tensor(0.0, dtype=self.dtype)

        y = self.read_value()

        if self.prior_on == PriorOn.CONSTRAINED:
            # evaluation is in same space as prior
            return tf.reduce_sum(self.prior.log_prob(y))

        else:
            # prior on unconstrained, but evaluating log-prior in constrained space
            x = self._unconstrained
            log_p = tf.reduce_sum(self.prior.log_prob(x))

            if self.transform is not None:
                # need to include log|Jacobian| to account for coordinate transform
                log_det_jacobian = self.transform.inverse_log_det_jacobian(y, y.shape.ndims)
                log_p += tf.reduce_sum(log_det_jacobian)

            return log_p

    @property
    def prior_on(self) -> PriorOn:
        return self._prior_on

    @prior_on.setter
    def prior_on(self, value: Union[str, PriorOn]) -> None:
        self._prior_on = PriorOn(value)

    def value(self) -> tf.Tensor:
        return _to_constrained(self._unconstrained.value(), self.transform)  # type: ignore  # assumes _to_constrained returns a tf.Tensor

    def read_value(self) -> tf.Tensor:
        return _to_constrained(self._unconstrained.read_value(), self.transform)  # type: ignore  # assumes _to_constrained returns a tf.Tensor

    def experimental_ref(self) -> "Parameter":
        return self

    def deref(self) -> "Parameter":
        return self

    @property
    def unconstrained_variable(self) -> tf.Variable:
        return self._unconstrained

    @property
    def transform(self) -> Optional[Transform]:
        return self._transform

    @transform.setter
    def transform(self, new_transform: Optional[Transform]) -> None:
        constrained_value = self.read_value()
        self._transform = new_transform
        self.assign(constrained_value)

    @property
    def trainable(self) -> bool:
        """
        `True` if this instance is trainable, else `False`.

        This attribute cannot be set directly. Use :func:`gpflow.set_trainable`.
        """
        return self._unconstrained.trainable

    @property
    def initial_value(self) -> tf.Tensor:
        return self._unconstrained.initial_value

    def validate_unconstrained_value(self, value: TensorData, dtype: DType) -> tf.Tensor:
        value = _cast_to_dtype(value, dtype)
        unconstrained_value = _to_unconstrained(value, self.transform)
        message = (
            "gpflow.Parameter: the value to be assigned is incompatible with this parameter's "
            "transform (the corresponding unconstrained value has NaN or Inf) and hence cannot be "
            "assigned."
        )
        return tf.debugging.assert_all_finite(unconstrained_value, message=message)

    def assign(
        self,
        value: TensorData,
        use_locking: bool = False,
        name: Optional[str] = None,
        read_value: bool = True,
    ) -> tf.Tensor:
        """
        Assigns constrained `value` to the unconstrained parameter's variable.
        It passes constrained value through parameter's transform first.

        Example:
            ```
            a = Parameter(2.0, transform=tfp.bijectors.Softplus())
            b = Parameter(3.0)

            a.assign(4.0)               # `a` parameter to `2.0` value.
            a.assign(tf.constant(5.0))  # `a` parameter to `5.0` value.
            a.assign(b)                 # `a` parameter to constrained value of `b`.
            ```

        :param value: Constrained tensor-like value.
        :param use_locking: If `True`, use locking during the assignment.
        :param name: The name of the operation to be created.
        :param read_value: if True, will return something which evaluates to the new
            value of the variable; if False will return the assign op.
        """
        unconstrained_value = self.validate_unconstrained_value(value, self.dtype)
        return self._unconstrained.assign(
            unconstrained_value, use_locking=use_locking, name=name, read_value=read_value
        )

    @property
    def is_tensor_like(self) -> bool:
        """
        This method means that TensorFlow's `tensor_util.is_tensor` function
        will return `True`
        """
        return True

    @property
    def name(self) -> str:
        return self._unconstrained.name

    @property
    def initializer(self):  # type unknown
        return self._unconstrained.initializer

    @property
    def device(self) -> Optional[str]:
        return self._unconstrained.device

    @property
    def dtype(self) -> tf.DType:
        return self._unconstrained.dtype

    @property
    def op(self) -> tf.Operation:
        return self._unconstrained.op

    @property
    def shape(self) -> tf.TensorShape:
        if self.transform is not None:
            return self.transform.forward_event_shape(self._unconstrained.shape)
        return self._unconstrained.shape

    def numpy(self) -> np.ndarray:
        return self.read_value().numpy()

    def get_shape(self) -> tf.TensorShape:
        return self.shape

    def _should_act_as_resource_variable(self):  # type unknown
        # needed so that Parameters are correctly identified by TensorFlow's
        # is_resource_variable() in resource_variable_ops.py
        pass  # only checked by TensorFlow using hasattr()

    @property
    def handle(self):  # type unknown
        return self._unconstrained.handle

    def __repr__(self) -> str:
        unconstrained = self.unconstrained_variable
        constrained = self.read_value()
        if tf.executing_eagerly():
            info = (
                f"unconstrained-shape={unconstrained.shape} "
                f"unconstrained-value={unconstrained.numpy()} "
                f"constrained-shape={constrained.shape} "
                f"constrained-value={constrained.numpy()}"
            )
        else:
            if unconstrained.shape == constrained.shape:
                info = f"shape={constrained.shape}"
            else:
                info = (
                    f"unconstrained-shape={unconstrained.shape} "
                    f"constrained-shape={constrained.shape}"
                )

        return f"<gpflow.Parameter {self.name!r} dtype={self.dtype.name} {info}>"

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


def _cast_to_dtype(
    value: TensorData, dtype: Optional[DType] = None
) -> Union[tf.Tensor, tf.Variable]:
    if dtype is None:
        dtype = default_float()

    if tf.is_tensor(value):
        # NOTE(awav) TF2.2 resolves issue with cast.
        # From TF2.2, `tf.cast` can be used alone instead of this auxiliary function.
        # workaround for https://github.com/tensorflow/tensorflow/issues/35938
        return tf.cast(value, dtype)
    else:
        return tf.convert_to_tensor(value, dtype=dtype)


def _to_constrained(value: TensorType, transform: Optional[Transform]) -> TensorType:
    if transform is not None:
        return transform.forward(value)
    return value


def _to_unconstrained(value: TensorType, transform: Optional[Transform]) -> TensorType:
    if transform is not None:
        return transform.inverse(value)
    return value
