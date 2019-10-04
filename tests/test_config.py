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

import gpflow
import numpy as np
import tensorflow as tf
import pytest
from gpflow import config


@pytest.mark.parametrize('getter, setter, valid_type_1, valid_type_2', [
    (config.default_int, config.set_default_int, np.int32, tf.int64),
    (config.default_float, config.set_default_float, np.float64, tf.float32),
])
def test_dtype_setting(getter, setter, valid_type_1, valid_type_2):
    if valid_type_1 == valid_type_2:
        raise ValueError("cannot test config setting/getting when both types are equal")
    setter(valid_type_1)
    assert getter() == valid_type_1
    setter(valid_type_2)
    assert getter() == valid_type_2


@pytest.mark.parametrize('setter, invalid_type', [
    (config.set_default_int, str),
    (config.set_default_int, float),
    (config.set_default_float, list),
    (config.set_default_float, int),
])
def test_dtype_errorcheck(setter, invalid_type):
    with pytest.raises(TypeError):
        setter(invalid_type)


def test_jitter_setting():
    config.set_default_jitter(1e-3)
    assert config.default_jitter() == 1e-3
    config.set_default_jitter(1e-6)
    assert config.default_jitter() == 1e-6


def test_jitter_errorcheck():
    with pytest.raises(TypeError):
        config.set_default_jitter("not a float")
    with pytest.raises(ValueError):
        config.set_default_jitter(-1e-10)


def test_summary_fmt_setting():
    config.set_summary_fmt("html")
    assert config.summary_fmt() == "html"
    config.set_summary_fmt(None)
    assert config.summary_fmt() == None


def test_summary_fmt_errorcheck():
    with pytest.raises(ValueError):
        config.set_summary_fmt("this_format_definitely_does_not_exist")
