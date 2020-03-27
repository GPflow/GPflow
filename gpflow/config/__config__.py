"""
This is a private module that manages GPflow configuration.

The module provides functions to modify default settings of GPflow, such as:
- the standard float precision and integer type
- the type of positive transformation
- a value for a minimum shift from zero for the positive transformation
- an output format for `gpflow.utilities.print_summary`

The module holds global configuration :class:`Config` variable that stores all
setting values.

Environment variables are an alternative way for changing the default GPflow
configuration.

.. warning::
    The user has to set environment variables before running python
    interpreter to modify the configuration.

Full set of environment variables and available options:

* ``GPFLOW_INT``: "int16", "int32", or "int64"
* ``GPFLOW_FLOAT``: "float16", "float32", or "float64"
* ``GPFLOW_POSITIVE_BIJECTOR``: "exp" or "softplus"
* ``GPFLOW_POSITIVE_MINIMUM``: Any positive float number
* ``GPFLOW_SUMMARY_FMT``: "notebook" or any other format that :mod:`tabulate` can handle.
* ``GPFLOW_JITTER``: Any positive float number

The user can also change the GPflow configuration temporarily with a context
manager :func:`as_context`:

>>> config = Config(jitter=1e-5)
>>> with as_context(config):
>>>     # ...code here sees new config
"""

import contextlib
import enum
import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional

import numpy as np
import tabulate
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    "Config",
    "as_context",
    "config",
    "set_config",
    "default_float",
    "set_default_float",
    "default_int",
    "set_default_int",
    "default_jitter",
    "set_default_jitter",
    "default_positive_bijector",
    "set_default_positive_bijector",
    "default_positive_minimum",
    "set_default_positive_minimum",
    "default_summary_fmt",
    "set_default_summary_fmt",
    "positive_bijector_type_map",
]

__config = None


class _Values(enum.Enum):
    """Setting's names collection with default values. The `name` method returns name
    of the environment variable. E.g. for `SUMMARY_FMT` field the environment variable
    will be `GPFLOW_SUMMARY_FMT`."""

    INT = np.int32
    FLOAT = np.float64
    POSITIVE_BIJECTOR = "softplus"
    POSITIVE_MINIMUM = None
    SUMMARY_FMT = "fancy_grid"
    JITTER = 1e-6

    @property
    def name(self):
        return f"GPFLOW_{super().name}"


def _default(value: _Values):
    """Checks if value is set in the environment."""
    return os.getenv(value.name, default=value.value)


def _default_numeric_type_factory(valid_types, enum_key, type_name):
    value = _default(enum_key)
    if value in valid_types.values():
        return value
    if value not in valid_types:
        raise TypeError(f"Config cannot recognize {type_name} type.")
    return valid_types[value]


def _default_int_factory():
    valid_types = dict(int16=np.int16, int32=np.int32, int64=np.int64)
    return _default_numeric_type_factory(valid_types, _Values.INT, "int")


def _default_float_factory():
    valid_types = dict(float16=np.float16, float32=np.float32, float64=np.float64)
    return _default_numeric_type_factory(valid_types, _Values.FLOAT, "float")


def _default_jitter_factory():
    try:
        value = float(_default(_Values.JITTER))
    except ValueError:
        raise TypeError("Config cannot set the jitter value with non float type.")
    return value


def _default_positive_bijector_factory():
    bijector_type = _default(_Values.POSITIVE_BIJECTOR)
    if bijector_type not in positive_bijector_type_map().keys():
        raise TypeError(
            "Config cannot set the passed value as a default positive bijector."
            f"Available options: {set(positive_bijector_type_map().keys())}"
        )
    return bijector_type


def _default_positive_minimum_factory():
    try:
        value = _default(_Values.POSITIVE_MINIMUM)
        if value is not None:
            value = float(value)
    except ValueError:
        raise TypeError("Config cannot set the positive_minimum value with non float type.")
    return value


def _default_summary_fmt_factory():
    return _default(_Values.SUMMARY_FMT)


@dataclass(frozen=True)
class Config:
    """
    Immutable object for storing global GPflow settings

    Args:
        int: Integer data type, int32 or int64.
        float: Float data type, float32 or float64
        jitter: Jitter value. Mainly used for for making badly conditioned matrices more stable.
            Default value is `1e-6`.
        positive_bijector: Method for positive bijector, either "softplus" or "exp".
            Default is "softplus".
        positive_minimum: Lower level for the positive transformation.
        summary_fmt: Summary format for module printing.
    """

    int: type = field(default_factory=_default_int_factory)
    float: type = field(default_factory=_default_float_factory)
    jitter: float = field(default_factory=_default_jitter_factory)
    positive_bijector: str = field(default_factory=_default_positive_bijector_factory)
    positive_minimum: float = field(default_factory=_default_positive_minimum_factory)
    summary_fmt: str = field(default_factory=_default_summary_fmt_factory)


def config() -> Config:
    """Returns current active config."""
    return __config


def default_int():
    """Returns default integer type"""
    return config().int


def default_float():
    """Returns default float type"""
    return config().float


def default_jitter():
    """
    The jitter is a constant that GPflow adds to the diagonal of matrices
    to achieve numerical stability of the system when the condition number 
    of the associated matrices is large, and therefore the matrices nearly singular.
    """
    return config().jitter


def default_positive_bijector():
    """Type of bijector used for positive constraints: exp or softplus."""
    return config().positive_bijector


def default_positive_minimum():
    """Shift constant that GPflow adds to all positive constraints."""
    return config().positive_minimum


def default_summary_fmt():
    """Summary printing format as understood by :mod:`tabulate` or a special case "notebook"."""
    return config().summary_fmt


def set_config(new_config: Config):
    """Update GPflow config with new settings from `new_config`."""
    global __config
    __config = new_config


def set_default_int(value_type):
    """
    Sets default integer type. Available options are ``np.int16``, ``np.int32``,
    or ``np.int64``.
    """
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_integer:
        raise TypeError(f"{value_type} is not an integer dtype")

    set_config(replace(config(), int=tf_dtype.as_numpy_dtype))


def set_default_float(value_type):
    """
    Sets default float type. Available options are `np.float16`, `np.float32`,
    or `np.float64`.
    """
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_floating:
        raise TypeError(f"{value_type} is not a float dtype")

    set_config(replace(config(), float=tf_dtype.as_numpy_dtype))


def set_default_jitter(value: float):
    """
    Sets constant jitter value.
    The jitter is a constant that GPflow adds to the diagonal of matrices
    to achieve numerical stability of the system when the condition number 
    of the associated matrices is large, and therefore the matrices nearly singular.
    """
    if not (
        isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0
    ) and not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Jitter must be non-negative")

    set_config(replace(config(), jitter=value))


def set_default_positive_bijector(value: str):
    """
    Sets positive bijector type.
    There are currently two options implemented: "exp" and "softplus".
    """
    type_map = positive_bijector_type_map()
    if isinstance(value, str):
        value = value.lower()
    if value not in type_map:
        raise ValueError(f"`{value}` not in set of valid bijectors: {sorted(type_map)}")

    set_config(replace(config(), positive_bijector=value))


def set_default_positive_minimum(value: float):
    """Sets shift constant for postive transformation."""
    if not (
        isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0
    ) and not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Value must be non-negative")

    set_config(replace(config(), positive_minimum=value))


def set_default_summary_fmt(value: str):
    formats = tabulate.tabulate_formats + ["notebook", None]
    if value not in formats:
        raise ValueError(f"Summary does not support '{value}' format")

    set_config(replace(config(), summary_fmt=value))


def positive_bijector_type_map() -> Dict[str, type]:
    return {
        "exp": tfp.bijectors.Exp,
        "softplus": tfp.bijectors.Softplus,
    }


@contextlib.contextmanager
def as_context(temporary_config: Optional[Config] = None):
    """Ensure that global configs defaults, with a context manager. Useful for testing."""
    current_config = config()
    temporary_config = replace(current_config) if temporary_config is None else temporary_config
    try:
        set_config(temporary_config)
        yield
    finally:
        set_config(current_config)


# Set global config.
set_config(Config())
