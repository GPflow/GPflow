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

import copy
import os
import tempfile

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow as gp
from gpflow.test_util import session_context, session_tf

# ==============================
# Fixtures and data definitions.
# ==============================


class Data:
    @staticmethod
    def deep_structure():
        a = gp.Param(1)
        b = gp.Param(2)
        c_a =  gp.Param(3)
        c_b =  gp.Param(4)

        with gp.defer_build():
            p =  gp.Parameterized()
            p.c =  gp.Parameterized()
            p.c.c =  gp.Parameterized()
            p.c.c.a =  gp.Param(3)
            p.c.c.b =  gp.Param(4)
            p.a = a
            p.b = b
            p.c.a = c_a
            p.c.b = c_b

        p.compile()
        return p

    @staticmethod
    def model():
        x = Data.x_new()
        y = np.random.rand(10, 1)
        kernel = gp.kernels.RBF(2)
        m = gp.models.GPR(x, y, kernel)
        return m

    @staticmethod
    def x_new():
        return np.random.rand(10, 2)


@pytest.fixture
def filename(request):
    with tempfile.NamedTemporaryFile() as file:
        yield file.name


@pytest.fixture
def deep_structure(session_tf):
    return Data.deep_structure()


@pytest.fixture
def model():
    return Data.model()


def simple_type_values():
    return [0, 0., 10, 10.0, np.array(1),
            np.array(10.), np.float16(2),
            np.int(10), np.float32(10),
            "test", "", None, True, False,
            tf.exp]


def list_type_values():
    return [
        [1, 2, 3],
        ["", "artem", 1, np.float(32), False],
        ["", 1, np.array([1,2,3]), True, None],
        np.array([10]),
        np.array([10, 20, 30]),
        np.array([10, 20, 30], dtype=np.float32),
    ]

def collection_type_values():
    return [
        {'a': 1, 'b': 'test', 'c': None},
        {'a': np.array([1])},
        {'a': np.array([1, 2, 3])},
        {'a': np.array([1, 2, 3]), 'b': ""}
    ]

# ======
# Tests.
# ======


@pytest.mark.parametrize('value', simple_type_values())
def test_encode_decode_simple_types(value):
    d = encode_decode(value)
    assert value == d


@pytest.mark.parametrize('value', list_type_values())
def test_encode_decode_list_types(value):
    d = encode_decode(value)
    def equal(x):
        a, b = x
        eq = a == b
        if isinstance(eq, np.ndarray) and eq.shape:
            return all(eq)
        return eq
    assert all(list(map(equal, zip(value, d))))


@pytest.mark.parametrize('value', collection_type_values())
def test_encode_decode_collection_types(value):
    d = encode_decode(value)
    assert value.keys() == d.keys()
    for k, v in value.items():
        if isinstance(v, np.ndarray) and v.shape:
            assert all(d[k] == v)
        else:
            assert d[k] == v


def test_saving_deep_parameterized_object(session_tf, filename, deep_structure):
    sess_a = session_tf
    gp.Saver().save(filename, deep_structure)
    with session_context() as sess_b:
        copy = gp.Saver().load(filename)
        equal_params(deep_structure.a, copy.a, session_a=sess_a, session_b=sess_b)
        equal_params(deep_structure.b, copy.b, session_a=sess_a, session_b=sess_b)
        equal_params(deep_structure.c.a, copy.c.a, session_a=sess_a, session_b=sess_b)
        equal_params(deep_structure.c.b, copy.c.b, session_a=sess_a, session_b=sess_b)
        equal_params(deep_structure.c.c.a, copy.c.c.a, session_a=sess_a, session_b=sess_b)
        equal_params(deep_structure.c.c.b, copy.c.c.b, session_a=sess_a, session_b=sess_b)


def test_saving_gpflow_model(session_tf, filename, model):
    x_new = Data.x_new()
    predict_origin = model.predict_f(x_new)
    gp.Saver().save(filename, model)
    with session_context() as session:
        loaded = gp.Saver().load(filename)
        predict_loaded = loaded.predict_f(x_new)
        assert_allclose(predict_origin, predict_loaded)


def test_loading_without_autocompile(session_tf, filename, model):
    gp.Saver().save(filename, model)
    with session_context() as session:
        context = gp.SaverContext(autocompile=False)
        loaded = gp.Saver().load(filename, context=context)
        assert loaded.is_built(session_tf.graph) == gp.Build.NO
        assert loaded.is_built(session.graph) == gp.Build.NO
        assert not any(loaded.trainable_tensors)


def test_loading_into_specific_session(session_tf, filename, model):
    x_new = Data.x_new()
    predict_origin = model.predict_f(x_new)
    gp.Saver().save(filename, model)
    with session_context() as session:
        context = gp.SaverContext(session=session)
        loaded = gp.Saver().load(filename, context=context)
        predict_loaded = loaded.predict_f(x_new, session=session)
    assert_allclose(predict_origin, predict_loaded)


# ========
# Helpers.
# ========


def encode_decode(value):
    ctx = gp.SaverContext()
    e = gp.saver.CoderDispatcher(ctx).encode(value)
    return gp.saver.CoderDispatcher(ctx).decode(e)


def equal_params(a, b, session_a=None, session_b=None):
    assert a.name == b.name
    assert a.pathname == b.pathname
    assert a.tf_pathname == b.tf_pathname
    assert a.tf_name_scope == b.tf_name_scope
    val_a = a.read_value(session=session_a)
    val_b = b.read_value(session=session_b)
    val_a_back = a.transform.backward(val_a)
    val_b_back = b.transform.backward(val_b)
    assert_allclose(val_a_back, val_b_back)
