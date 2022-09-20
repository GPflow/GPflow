# Copyright 2019-2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from typing import Any, Dict, Generator, List, Mapping, Optional, Union

import numpy as np
import tabulate
import tensorflow as tf
import tensorflow_probability as tfp

__all__ = [
    "Config",
    "as_context",
    "config",
    "default_float",
    "default_int",
    "default_jitter",
    "default_positive_bijector",
    "default_positive_minimum",
    "default_summary_fmt",
    "positive_bijector_type_map",
    "set_config",
    "set_default_float",
    "set_default_int",
    "set_default_jitter",
    "set_default_positive_bijector",
    "set_default_positive_minimum",
    "set_default_summary_fmt",
]

__config: Optional["Config"] = None


class _Values(enum.Enum):
    """Setting's names collection with default values. The `name` method returns name
    of the environment variable. E.g. for `SUMMARY_FMT` field the environment variable
    will be `GPFLOW_SUMMARY_FMT`."""

    INT = np.int32
    FLOAT = np.float64
    POSITIVE_BIJECTOR = "softplus"
    POSITIVE_MINIMUM = 0.0
    SUMMARY_FMT = "fancy_grid"
    JITTER = 1e-6

    @property
    def name(self) -> str:  # type: ignore[override]  # name is generated and has weird typing.
        return f"GPFLOW_{super().name}"


def _default(value: _Values) -> Any:
    """Checks if value is set in the environment."""
    return os.getenv(value.name, default=value.value)


def _default_numeric_type_factory(
    valid_types: Mapping[str, type], enum_key: _Values, type_name: str
) -> type:
    value: Union[str, type] = _default(enum_key)
    if isinstance(value, type) and (value in valid_types.values()):
        return value
    assert isinstance(value, str)  # Hint for mypy
    if value not in valid_types:
        raise TypeError(f"Config cannot recognize {type_name} type.")
    return valid_types[value]


def _default_int_factory() -> type:
    valid_types = dict(int16=np.int16, int32=np.int32, int64=np.int64)
    return _default_numeric_type_factory(valid_types, _Values.INT, "int")


def _default_float_factory() -> type:
    valid_types = dict(float16=np.float16, float32=np.float32, float64=np.float64)
    return _default_numeric_type_factory(valid_types, _Values.FLOAT, "float")


def _default_jitter_factory() -> float:
    value = _default(_Values.JITTER)
    try:
        return float(value)
    except ValueError:
        raise TypeError("Config cannot set the jitter value with non float type.")


def _default_positive_bijector_factory() -> str:
    bijector_type: str = _default(_Values.POSITIVE_BIJECTOR)
    if bijector_type not in positive_bijector_type_map().keys():
        raise TypeError(
            "Config cannot set the passed value as a default positive bijector."
            f"Available options: {set(positive_bijector_type_map().keys())}"
        )
    return bijector_type


def _default_positive_minimum_factory() -> float:
    value = _default(_Values.POSITIVE_MINIMUM)
    try:
        return float(value)
    except ValueError:
        raise TypeError("Config cannot set the positive_minimum value with non float type.")


def _default_summary_fmt_factory() -> Optional[str]:
    result: Optional[str] = _default(_Values.SUMMARY_FMT)
    return result


# The following type alias is for the Config class, to help a static analyser distinguish
# between the built-in 'float' type and the 'float' type defined in the that class.
Float = Union[float]


@dataclass(frozen=True)
class Config:
    """
    Immutable object for storing global GPflow settings
    """

    int: type = field(default_factory=_default_int_factory)
    """Integer data type, int32 or int64."""

    float: type = field(default_factory=_default_float_factory)
    """Float data type, float32 or float64"""

    jitter: Float = field(default_factory=_default_jitter_factory)
    """
    Jitter value. Mainly used for for making badly conditioned matrices more stable.

    Default value is `1e-6`.
    """

    positive_bijector: str = field(default_factory=_default_positive_bijector_factory)
    """
    Method for positive bijector, either "softplus" or "exp".

    Default is "softplus".
    """

    positive_minimum: Float = field(default_factory=_default_positive_minimum_factory)
    """Lower bound for the positive transformation."""

    summary_fmt: Optional[str] = field(default_factory=_default_summary_fmt_factory)
    """Summary format for module printing."""


def config() -> Config:
    """Returns current active config."""
    assert __config is not None, "__config is None. This should never happen."
    return __config


def default_int() -> type:
    """Returns default integer type"""
    return config().int


def default_float() -> type:
    """Returns default float type"""
    return config().float


def default_jitter() -> float:
    """
    The jitter is a constant that GPflow adds to the diagonal of matrices
    to achieve numerical stability of the system when the condition number
    of the associated matrices is large, and therefore the matrices nearly singular.
    """
    return config().jitter


def default_positive_bijector() -> str:
    """Type of bijector used for positive constraints: exp or softplus."""
    return config().positive_bijector


def default_positive_minimum() -> float:
    """Shift constant that GPflow adds to all positive constraints."""
    return config().positive_minimum


def default_summary_fmt() -> Optional[str]:
    """Summary printing format as understood by :mod:`tabulate` or a special case "notebook"."""
    return config().summary_fmt


def set_config(new_config: Config) -> None:
    """Update GPflow config with new settings from `new_config`."""
    global __config
    __config = new_config


def set_default_int(value_type: type) -> None:
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


def set_default_float(value_type: type) -> None:
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


def set_default_jitter(value: float) -> None:
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


def set_default_positive_bijector(value: str) -> None:
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


def set_default_positive_minimum(value: float) -> None:
    """Sets shift constant for positive transformation."""
    if not (
        isinstance(value, (tf.Tensor, np.ndarray)) and len(value.shape) == 0
    ) and not isinstance(value, float):
        raise TypeError("Expected float32 or float64 scalar value")

    if value < 0:
        raise ValueError("Positive minimum must be non-negative")

    set_config(replace(config(), positive_minimum=value))


def set_default_summary_fmt(value: Optional[str]) -> None:
    formats: List[Optional[str]] = list(tabulate.tabulate_formats)
    formats.extend(["notebook", None])
    if value not in formats:
        raise ValueError(f"Summary does not support '{value}' format")

    set_config(replace(config(), summary_fmt=value))


def positive_bijector_type_map() -> Dict[str, type]:
    return {
        "exp": tfp.bijectors.Exp,
        "softplus": tfp.bijectors.Softplus,
    }


@contextlib.contextmanager
def as_context(temporary_config: Optional[Config] = None) -> Generator[None, None, None]:
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
