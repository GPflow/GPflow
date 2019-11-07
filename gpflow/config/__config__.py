import contextlib
import enum
import os
from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import tabulate
import tensorflow as tf

__all__ = [
    "summary_fmt", "default_float", "default_jitter", "default_int", "set_summary_fmt", "set_default_float",
    "set_default_jitter", "set_default_int", "as_context", "config", "set_config", "positive_minimum",
    "set_positive_minimum"
]

__config = None


class _Values(enum.Enum):
    INT = np.int32
    FLOAT = np.float64
    POSITIVE_MINIMUM = None
    JITTER = 1e-6
    SUMMARY_FMT = None

    @property
    def name(self):
        return f"GPFLOW_{super().name}"


def default(value: _Values):
    return os.getenv(value.name, default=value.value)


@dataclass(frozen=True)
class Config:
    int: type = field(default_factory=lambda: default(_Values.INT))
    float: type = field(default_factory=lambda: default(_Values.FLOAT))
    jitter: float = field(default_factory=lambda: default(_Values.JITTER))
    positive_minimum: float = field(default_factory=lambda: default(_Values.POSITIVE_MINIMUM))
    summary_fmt: str = _Values.SUMMARY_FMT.value


def config() -> Config:
    return __config


def summary_fmt():
    return config().summary_fmt


def default_int():
    return config().int


def default_float():
    return config().float


def default_jitter():
    return config().jitter


def positive_minimum():
    return config().positive_minimum


def set_config(new_config: Config):
    global __config
    __config = new_config


def set_default_int(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_integer:
        raise TypeError(f"{value_type} is not an integer dtype")

    set_config(replace(config(), int=value_type))


def set_default_float(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")

    if not tf_dtype.is_floating:
        raise TypeError(f"{value_type} is not a float dtype")

    set_config(replace(config(), float=value_type))


def set_default_jitter(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Jitter must be non-negative")

    set_config(replace(config(), jitter=value))


def set_summary_fmt(value: str):
    formats = tabulate.tabulate_formats + ['notebook', None]
    if value not in formats:
        raise ValueError(f"Summary does not support '{value}' format")

    set_config(replace(config(), summary_fmt=value))


def set_positive_minimum(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Jitter must be non-negative")

    set_config(replace(config(), jitter=value))


@contextlib.contextmanager
def as_context(temporary_config: Optional[Config] = None):
    '''Ensure that global configs defaults, with a context manager. Useful
    for testing.'''
    current_config = config()
    temporary_config = replace(current_config) if temporary_config is None else temporary_config
    try:
        set_config(temporary_config)
        yield
    finally:
        set_config(current_config)


# Set global config.
set_config(Config())
