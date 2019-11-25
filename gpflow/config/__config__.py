import contextlib
import enum
import os
from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Union

import numpy as np
import tabulate
import tensorflow as tf
import tensorflow_probability as tfp


__all__ = [
    "Config", "as_context", "config", "set_config",
    "default_float", "set_default_float",
    "default_int", "set_default_int",
    "default_jitter", "set_default_jitter",
    "default_positive_bijector", "set_default_positive_bijector",
    "default_positive_minimum", "set_default_positive_minimum",
    "default_summary_fmt", "set_default_summary_fmt",
    "positive_bijector_type_map"
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
    SUMMARY_FMT = None
    JITTER = 1e-6

    @property
    def name(self):
        return f"GPFLOW_{super().name}"


def default(value: _Values):
    """Checks if value is set in the environment."""
    return os.getenv(value.name, default=value.value)


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

    int: type = field(default_factory=lambda: default(_Values.INT))
    float: type = field(default_factory=lambda: default(_Values.FLOAT))
    jitter: float = field(default_factory=lambda: default(_Values.JITTER))
    positive_bijector: str = field(default_factory=lambda: default(_Values.POSITIVE_BIJECTOR))
    positive_minimum: float = field(default_factory=lambda: default(_Values.POSITIVE_MINIMUM))
    summary_fmt: str = field(default_factory=lambda: default(_Values.SUMMARY_FMT))


def config() -> Config:
    """Returns current active config."""
    return __config


def default_int():
    return config().int


def default_float():
    return config().float


def default_jitter():
    return config().jitter


def default_positive_bijector():
    return config().positive_bijector


def default_positive_minimum():
    return config().positive_minimum


def default_summary_fmt():
    return config().summary_fmt


def set_config(new_config: Config):
    """Update GPflow config"""
    global __config
    __config = new_config


def set_default_int(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_integer:
        raise TypeError(f"{value_type} is not an integer dtype")

    set_config(replace(config(), int=tf_dtype.as_numpy_dtype))


def set_default_float(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_floating:
        raise TypeError(f"{value_type} is not a float dtype")

    set_config(replace(config(), float=tf_dtype.as_numpy_dtype))


def set_default_jitter(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Jitter must be non-negative")

    set_config(replace(config(), jitter=value))


def set_default_positive_bijector(value: str):
    type_map = positive_bijector_type_map()
    if isinstance(value, str):
        value = value.lower()
    if value not in type_map:
        raise ValueError(f"`{value}` not in set of valid bijectors: {sorted(type_map)}")

    set_config(replace(config(), positive_bijector=value))


def set_default_positive_minimum(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Value must be non-negative")

    set_config(replace(config(), positive_minimum=value))


def set_default_summary_fmt(value: str):
    formats = tabulate.tabulate_formats + ['notebook', None]
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
