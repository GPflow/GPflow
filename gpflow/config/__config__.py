import contextlib
import enum
import os
from dataclasses import dataclass, field

import numpy as np
import tabulate
import tensorflow as tf

__all__ = [
    "summary_fmt", "default_float", "default_jitter", "default_int", "set_summary_fmt", "set_default_float",
    "set_default_jitter", "set_default_int", "context"
]


class _Values(enum.Enum):
    INT = np.int32
    FLOAT = np.float64
    POSITIVE_MINIMAL = 1e-6
    JITTER = 1e-6
    SUMMARY_FMT = "pretty_grid"

    @property
    def name(self):
        return f"GPFLOW_{super().name}"


def default(value: _Values):
    return os.getenv(value.name, default=value.value)


@dataclass(frozen=True)
class _Config:
    int: type = field(default_factory=lambda: default(_Values.INT))
    float: type = field(default_factory=lambda: default(_Values.FLOAT))
    jitter: float = field(default_factory=lambda: default(_Values.JITTER))
    positive_minimal: float = field(default_factory=lambda: default(_Values.POSITIVE_MINIMAL))
    summary_fmt: str = _Values.SUMMARY_FMT.value


__config = _Config()


def summary_fmt():
    return __config._summary_fmt


def default_int():
    return __config._int


def default_float():
    return __config._float


def default_jitter():
    return __config._jitter


def positive_minimum_value():
    return __config._positive_minimum_value


def set_default_int(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")
    if not tf_dtype.is_integer:
        raise TypeError(f"{value_type} is not an integer dtype")
    __config._int = value_type


def set_default_float(value_type):
    try:
        tf_dtype = tf.as_dtype(value_type)  # Test that it's a tensorflow-valid dtype
    except TypeError:
        raise TypeError(f"{value_type} is not a valid tf or np dtype")
    if not tf_dtype.is_floating:
        raise TypeError(f"{value_type} is not a float dtype")
    __config._float = value_type


def set_default_jitter(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")
    if value < 0:
        raise ValueError("Jitter must be non-negative")

    __config._jitter = value


def set_summary_fmt(fmt: str):
    formats = tabulate.tabulate_formats + ['notebook', None]
    if fmt not in formats:
        raise ValueError(f"Summary does not support '{fmt}' format")

    __config._summary_fmt = fmt


@contextlib.contextmanager
def context():
    '''Ensure that global configs defaults, with a context manager. Useful
    for testing.'''
    float_dtype = default_float()
    int_dtype = default_int()
    jitter = default_jitter()
    fmt = summary_fmt()
    yield
    set_default_float(float_dtype)
    set_default_int(int_dtype)
    set_default_jitter(jitter)
    set_summary_fmt(fmt)
