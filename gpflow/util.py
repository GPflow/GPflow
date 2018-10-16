from typing import Optional, Union, List, Tuple
import logging
import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
import tensorflow_probability as tfp


def create_logger(name=None):
    return logging.getLogger('Temporary Logger Solution')


def jitter_eye(num_rows: int, num_columns: int = None, value: float = None) -> float:
    value = default_jitter() if value is None else value
    return tf.eye(num_rows, num_columns=num_columns, dtype=default_float()) * value


def default_jitter() -> float:
    return 1e-6


def default_float() -> float:
    return np.float64


def default_int() -> int:
    return np.int32