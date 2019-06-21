from typing import TypeVar
import os
from dataclasses import dataclass, field

import numpy as np
import tensorflow as tf

__all__ = ["default_float", "default_jitter"]


def _int_from_env(env_name: str = "GPFLOW_INT", default=np.int32):
    return tf.as_dtype(os.getenv(env_name, default=default))


def _float_from_env(env_name: str = "GPFLOW_FLOAT", default=np.float32):
    return tf.as_dtype(os.getenv(env_name, default=default))


def _jitter_from_env(env_name: str = "GPFLOW_JITTER", default=1e-6):
    return float(os.getenv(env_name, default=default))


@dataclass
class _Config:
    int: tf.DType = field(default_factory=_int_from_env)
    float: tf.DType = field(default_factory=_float_from_env)
    jitter: float = field(default_factory=_jitter_from_env)


__config = _Config()


def default_int():
    return __config.int


def default_float():
    return __config.float


def default_jitter():
    return __config.jitter


def set_default_int(value_type):
    try:
        tf.as_dtype(value_type)  # Test input value that it is eligable type.
        __config.int = value_type
    except TypeError:
        raise TypeError("Expected tf or np dtype argument")


def set_default_float(value_type):
    try:
        tf.as_dtype(value_type)  # Test input value that it is eligable type.
        __config.float = value_type
    except TypeError:
        raise TypeError("Expected tf or np dtype argument")


def set_default_jitter(value: float):
    if not (isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0) and \
            not isinstance(value, float):
        raise ValueError("Expected float32 or float64 scalar value")

    __config.jitter = value
