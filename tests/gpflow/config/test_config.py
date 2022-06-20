# Copyright 2018 the GPflow authors.
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

import os
from typing import Any, Callable
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.base import TensorData
from gpflow.config import (
    default_float,
    default_int,
    default_jitter,
    default_positive_bijector,
    default_summary_fmt,
    set_default_float,
    set_default_int,
    set_default_jitter,
    set_default_positive_bijector,
    set_default_summary_fmt,
)
from gpflow.utilities import to_default_float, to_default_int

_env_values = [
    ("int", "int16", np.int16),
    ("int", "int64", np.int64),
    ("float", "float16", np.float16),
    ("float", "float32", np.float32),
    ("positive_bijector", "exp", "exp"),
    ("positive_bijector", "softplus", "softplus"),
    ("summary_fmt", "simple", "simple"),
    ("positive_minimum", "1e-3", 1e-3),
    ("jitter", "1e-2", 1e-2),
]


@pytest.mark.parametrize("attr_name, value, expected_value", _env_values)
def test_env_variables(attr_name: str, value: str, expected_value: Any) -> None:
    env_name = f"GPFLOW_{attr_name.upper()}"
    with mock.patch.dict("os.environ", {env_name: value}):
        assert os.environ[env_name] == value
        config = gpflow.config.Config()
        assert getattr(config, attr_name) == expected_value


@pytest.mark.parametrize("attr_name", dict.fromkeys(list(zip(*_env_values))[0]).keys())
def test_env_variables_failures(attr_name: str) -> None:
    if attr_name == "summary_fmt":
        pytest.skip("The `summary_fmt` validation cannot be performed.")
    env_name = f"GPFLOW_{attr_name.upper()}"
    with mock.patch.dict("os.environ", {env_name: "garbage"}):
        with pytest.raises(TypeError):
            gpflow.config.Config()


@pytest.mark.parametrize(
    "getter, setter, valid_type_1, valid_type_2",
    [
        (default_int, set_default_int, tf.int64, np.int32),
        (default_float, set_default_float, tf.float32, np.float64),
    ],
)
def test_dtype_setting(
    getter: Callable[[], type],
    setter: Callable[[type], None],
    valid_type_1: type,
    valid_type_2: type,
) -> None:
    if valid_type_1 == valid_type_2:
        raise ValueError("cannot test config setting/getting when both types are equal")
    setter(valid_type_1)
    assert getter() == valid_type_1
    setter(valid_type_2)
    assert getter() == valid_type_2


@pytest.mark.parametrize(
    "setter, invalid_type",
    [
        (set_default_int, str),
        (set_default_int, np.float64),
        (set_default_float, list),
        (set_default_float, tf.int32),
    ],
)
def test_dtype_errorcheck(setter: Callable[[type], None], invalid_type: Any) -> None:
    with pytest.raises(TypeError):
        setter(invalid_type)


def test_jitter_setting() -> None:
    set_default_jitter(1e-3)
    assert default_jitter() == 1e-3
    set_default_jitter(1e-6)
    assert default_jitter() == 1e-6


def test_jitter_errorcheck() -> None:
    with pytest.raises(TypeError):
        set_default_jitter("not a float")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        set_default_jitter(-1e-10)


@pytest.mark.parametrize(
    "value, error_msg",
    [
        ("Unknown", r"`unknown` not in set of valid bijectors: \['exp', 'softplus'\]"),
        (1.0, r"`1.0` not in set of valid bijectors: \['exp', 'softplus'\]"),
    ],
)
def test_positive_bijector_error(value: Any, error_msg: str) -> None:
    with pytest.raises(ValueError, match=error_msg):
        set_default_positive_bijector(value)


@pytest.mark.parametrize("value", ["exp", "SoftPlus"])
def test_positive_bijector_setting(value: str) -> None:
    set_default_positive_bijector(value)
    assert default_positive_bijector() == value.lower()


def test_default_summary_fmt_setting() -> None:
    set_default_summary_fmt("html")
    assert default_summary_fmt() == "html"
    set_default_summary_fmt(None)
    assert default_summary_fmt() is None


def test_default_summary_fmt_errorcheck() -> None:
    with pytest.raises(ValueError):
        set_default_summary_fmt("this_format_definitely_does_not_exist")


@pytest.mark.parametrize(
    "setter, getter, converter, dtype, value",
    [
        (set_default_int, default_int, to_default_int, np.int32, 3),
        (set_default_int, default_int, to_default_int, tf.int32, 3),
        (set_default_int, default_int, to_default_int, tf.int64, [3, 1, 4, 1, 5, 9]),
        (set_default_int, default_int, to_default_int, np.int64, [3, 1, 4, 1, 5, 9]),
        (set_default_float, default_float, to_default_float, np.float32, 3.14159),
        (
            set_default_float,
            default_float,
            to_default_float,
            tf.float32,
            [3.14159, 3.14159, 3.14159],
        ),
        (
            set_default_float,
            default_float,
            to_default_float,
            np.float64,
            [3.14159, 3.14159, 3.14159],
        ),
        (
            set_default_float,
            default_float,
            to_default_float,
            tf.float64,
            [3.14159, 3.14159, 3.14159],
        ),
    ],
)
def test_native_to_default_dtype(
    setter: Callable[[type], None],
    getter: Callable[[], type],
    converter: Callable[[TensorData], tf.Tensor],
    dtype: type,
    value: TensorData,
) -> None:
    with gpflow.config.as_context():
        setter(dtype)
        assert converter(value).dtype == dtype
        assert converter(value).dtype == getter()
