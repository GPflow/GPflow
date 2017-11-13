# Copyright 2017 Artem Artemev @awav
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

import functools
import contextlib

import tensorflow as tf

from .core.base import GPflowError
from .core.base import Build
from .core.node import Node
from .core.autoflow import AutoFlow, TensorType
from .core.tensor_converter import TensorConverter

from .params import Parameterized


def name_scope(name=None):
    def name_scope_wrapper(method):
        @functools.wraps(method)
        def runnable(*args, **kwargs):
            scope_name = name if name is not None else method.__name__
            with tf.name_scope(scope_name):
                return method(*args, **kwargs)
        return runnable
    return name_scope_wrapper


def params_as_tensors(method):
    @functools.wraps(method)
    def tensor_mode_wrapper(obj, *args, **kwargs):
        if not isinstance(obj, Parameterized):
            raise GPflowError(
                'Tensor mode works only for parmeterized object.')
        prev_value = _params_as_tensors_enter(obj, True)
        try:
            result = method(obj, *args, **kwargs)
        finally:
            _params_as_tensors_exit(obj, prev_value)
        return result
    return tensor_mode_wrapper


@contextlib.contextmanager
def params_as_tensors_for(obj, convert=True):
    prev_value = _params_as_tensors_enter(obj, convert)
    try:
        yield
    finally:
        _params_as_tensors_exit(obj, prev_value)


def autoflow(*af_args, **af_kwargs):
    def autoflow_wrapper(method):
        @functools.wraps(method)
        def runnable(obj, *args, **kwargs):
            if not isinstance(obj, Node):
                raise GPflowError(
                    'Tensor mode works only for node-like object.')
            if obj.is_built_coherence(obj.graph) is Build.NO:
                raise GPflowError('Compilable object is not built.')
            name = method.__name__
            store = AutoFlow.get_autoflow(obj, name)
            session = kwargs.pop('session', None)
            session = obj.enquire_session(session=session)
            if not store:
                scope_name = _name_scope_name(obj, name)
                with session.graph.as_default(), tf.name_scope(scope_name):
                    _setup_storage(store, *af_args, **af_kwargs)
                    _build_method(method, obj, store)
            return _session_run(session, obj, store, *args, **kwargs)
        return runnable
    return autoflow_wrapper


def _params_as_tensors_enter(obj, convert=True):
    name = TensorConverter.__tensor_mode__
    attr_value = getattr(obj, name, None)
    setattr(obj, name, convert)
    return attr_value


def _params_as_tensors_exit(obj, previous):
    name = TensorConverter.__tensor_mode__
    if previous is not None:
        setattr(obj, name, previous)
    else:
        delattr(obj, name)


def _setup_storage(store, *args, **_kwargs):
    store['arguments'] = TensorType.make_structure(args)


def _name_scope_name(obj, name):
    return '/'.join(['autoflow', obj.name, name])


def _session_run(session, obj, store, *args, **kwargs):
    feed_dict_key = 'feed_dict'
    if feed_dict_key not in kwargs:
        kwargs[feed_dict_key] = {}
    feed_dict = kwargs.get(feed_dict_key)
    feed_dict.update(dict(zip(store['arguments'], args)))
    if obj.feeds:
        feed_dict.update(obj.feeds)
    return session.run(store['result'], **kwargs)


def _build_method(method, obj, store):
    store['result'] = method(obj, *store['arguments'])
