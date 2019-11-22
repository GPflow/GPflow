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

import numpy as np
import pytest
import tensorflow as tf
import tensorflow_probability as tfp

import gpflow
from gpflow.config import (default_float, default_int, default_jitter, set_default_float,
                           set_default_int, set_default_jitter, set_default_summary_fmt,
                           default_summary_fmt, set_default_positive_bijector,
                           default_positive_bijector)
from gpflow.utilities import to_default_float, to_default_int


@pytest.mark.parametrize('getter, setter, valid_type_1, valid_type_2', [
    (default_int, set_default_int, tf.int64, np.int32),
    (default_float, set_default_float, tf.float32, np.float64),
])
def test_dtype_setting(getter, setter, valid_type_1, valid_type_2):
    if valid_type_1 == valid_type_2:
        raise ValueError("cannot test config setting/getting when both types are equal")
    setter(valid_type_1)
    assert getter() == valid_type_1
    setter(valid_type_2)
    assert getter() == valid_type_2


@pytest.mark.parametrize('setter, invalid_type', [
    (set_default_int, str),
    (set_default_int, np.float64),
    (set_default_float, list),
    (set_default_float, tf.int32),
])
def test_dtype_errorcheck(setter, invalid_type):
    with pytest.raises(TypeError):
        setter(invalid_type)


def test_jitter_setting():
    set_default_jitter(1e-3)
    assert default_jitter() == 1e-3
    set_default_jitter(1e-6)
    assert default_jitter() == 1e-6


def test_jitter_errorcheck():
    with pytest.raises(TypeError):
        set_default_jitter("not a float")
    with pytest.raises(ValueError):
        set_default_jitter(-1e-10)


@pytest.mark.parametrize("value, error_msg", [
    ("Unknown", r"`unknown` not in set of valid bijectors: \['exp', 'softplus'\]"),
    (1.0, r"`1.0` not in set of valid bijectors: \['exp', 'softplus'\]"),
])
def test_positive_bijector_error(value, error_msg):
    with pytest.raises(ValueError, match=error_msg):
        set_default_positive_bijector(value)


@pytest.mark.parametrize("value", ["exp", "SoftPlus"])
def test_positive_bijector_setting(value):
    set_default_positive_bijector(value)
    assert default_positive_bijector() == value.lower()


def test_default_summary_fmt_setting():
    set_default_summary_fmt("html")
    assert default_summary_fmt() == "html"
    set_default_summary_fmt(None)
    assert default_summary_fmt() is None


def test_default_summary_fmt_errorcheck():
    with pytest.raises(ValueError):
        set_default_summary_fmt("this_format_definitely_does_not_exist")


@pytest.mark.parametrize('setter, getter, converter, dtype, value', [
    (set_default_int, default_int, to_default_int, np.int32, 3),
    (set_default_int, default_int, to_default_int, tf.int32, 3),
    (set_default_int, default_int, to_default_int, tf.int64, [3, 1, 4, 1, 5, 9]),
    (set_default_int, default_int, to_default_int, np.int64, [3, 1, 4, 1, 5, 9]),
    (set_default_float, default_float, to_default_float, np.float32, 3.14159),
    (set_default_float, default_float, to_default_float, tf.float32, [3.14159, 3.14159, 3.14159]),
    (set_default_float, default_float, to_default_float, np.float64, [3.14159, 3.14159, 3.14159]),
    (set_default_float, default_float, to_default_float, tf.float64, [3.14159, 3.14159, 3.14159]),
])
def test_native_to_default_dtype(setter, getter, converter, dtype, value):
    with gpflow.config.as_context():
        setter(dtype)
        assert converter(value).dtype == dtype
        assert converter(value).dtype == getter()
