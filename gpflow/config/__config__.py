import os

import numpy as np
import tensorflow as tf

__all__ = ["default_float", "default_jitter"]

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


__config = _Config()


def default_int():
    return __config._int


def default_float():
    return __config._float


def default_jitter():
    return __config._jitter


def set_default_int(value_type):
    try:
        tf.as_dtype(value_type)  # Test input value that it is eligable type.
        __config._int = value_type
    except TypeError:
        raise TypeError("Expected tf or np dtype argument")


def set_default_float(value_type):
    try:
        tf.as_dtype(value_type)  # Test input value that it is eligable type.
        __config._float = value_type
    except TypeError:
        raise TypeError("Expected tf or np dtype argument")


def set_default_jitter(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise ValueError("Expected float32 or float64 scalar value")

    __config._jitter = value
