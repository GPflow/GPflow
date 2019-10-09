import os

import numpy as np
import tabulate
import tensorflow as tf

__all__ = [
    "summary_fmt",
    "default_float",
    "default_jitter",
    "default_int",
    "set_summary_fmt",
    "set_default_float",
    "set_default_jitter",
    "set_default_int",
]

_ENV_JITTER = "GPFLOW_JITTER"
_ENV_FLOAT = "GPFLOW_FLOAT"
_ENV_INT = "GPFLOW_INT"

_JITTER_VALUE = 1e-6
_FLOAT_VALUE = np.float64
_INT_VALUE = np.int32


class _Config:
    def __init__(self):
        self._int = os.getenv(_ENV_INT, default=_INT_VALUE)
        self._float = os.getenv(_ENV_FLOAT, default=_FLOAT_VALUE)
        self._jitter = os.getenv(_ENV_JITTER, default=_JITTER_VALUE)
        self._summary_fmt = None


__config = _Config()


def summary_fmt():
    return __config._summary_fmt


def default_int():
    return __config._int


def default_float():
    return __config._float


def default_jitter():
    return __config._jitter


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
