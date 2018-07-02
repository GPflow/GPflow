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

from .core.errors import GPflowError
from .core.compilable import Build
from .core.compilable import AutoBuildStatus
from .core.node import Node
from .core.autoflow import AutoFlow
from .core.tensor_converter import TensorConverter

from .params import Parameterized


def name_scope(name=None):
    """
    This decorator wraps a function so that it runs inside a TensorFlow
    name scope. The name is given by the `name` option; if this is None,
    then the name of the function will be used.
    ```
    >>> @name_scope()
    >>> def foo(...):
    >>>     # now runs inside scope "foo"
    >>> @name_scope('bar')
    >>> def baz(...):
    >>>     # now runs inside scope "bar", not "baz"
    ```
    """
    def name_scope_wrapper_decorator(method):
        @functools.wraps(method)
        def name_scope_wrapper(*args, **kwargs):
            scope_name = name if name is not None else method.__name__
            with tf.name_scope(scope_name):
                return method(*args, **kwargs)
        return name_scope_wrapper
    return name_scope_wrapper_decorator


def params_as_tensors(method):
    """
    The `params_as_tensors` decorator converts representation for parameters into
    their unconstrained tensors, and data holders to their data tensors inside
    wrapped function, subject to this function is a member of parameterized object.
    """
    @functools.wraps(method)
    def tensor_mode_wrapper(obj, *args, **kwargs):
        if not isinstance(obj, Parameterized):
            raise GPflowError(
                'Tensor mode works only for Parameterized object.')
        prev_value = _params_as_tensors_enter(obj, True)
        try:
            result = method(obj, *args, **kwargs)
        finally:
            _params_as_tensors_exit(obj, prev_value)
        return result
    return tensor_mode_wrapper


class defer_build(contextlib.ContextDecorator):
    """
    The `defer_build` can be either context manager or decorator. In both cases it
    cancels autobuild feature for all gpflow ICompilable objects. Sometimes,
    it is require to build model with aligned names or for some other reasons.

    Example below shows that `defer_build` allows you to create kernel and
    change parameters in it without running into an exception that the model
    has already been changed.

    ```
    X = np.linspace(-3,3,20)
    Y = np.random.exponential(np.sin(X)**2)

    with gpflow.defer_build():
        k = gpflow.kernels.Matern32(1, ARD=False) + gpflow.kernels.Bias(1)
        l = gpflow.likelihoods.Exponential()
        m = gpflow.models.GPMC(X, Y, k, l)
        m.kern.matern32.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
        m.kern.matern32.variance.prior = gpflow.priors.Gamma(1., 1.)
        m.kern.bias.variance.prior = gpflow.priors.Gamma(1., 1.)

    ...
    m.compile()
    ```

    :param defer: Option to control defer mechanics. If `defer` is `False` AutoBuild
        feature will as usual.
    """

    def __init__(self, defer=True):
        self.defer = defer
        self.prev_autobuild_status = None

    def __enter__(self):
        self.prev_autobuild_status = AutoBuildStatus.__autobuild_enabled_global__
        AutoBuildStatus.__autobuild_enabled_global__ = not self.defer

    def __exit__(self, *exc):
        AutoBuildStatus.__autobuild_enabled_global__ = self.prev_autobuild_status
        return False


@contextlib.contextmanager
def params_as_tensors_for(*objs, convert=True):
    """
    Context manager which changes the representation of parameters and data holders
    for the specific parameterized object(s).

    This can also be used to turn off tensor conversion functions wrapped with
    `params_as_tensors`:
    ```
    @gpflow.params_as_tensors
    def compute_something(self):  # self is parameterized object.
        s = tf.reduce_sum(self.a) # self.a is a parameter.
        with params_as_tensors_for(self, convert=False):
            b = self.c.constrained_tensor
        return s + b
    ```

    :param objs: one or more instances of classes deriving from Parameterized
    :param convert: Flag which is used for turning tensor convertion
        feature on, `True`, or turning it off, `False`.
    """
    objs = set(objs)  # remove duplicate objects so the tensor mode won't be changed before saving
    prev_values = [_params_as_tensors_enter(o, convert) for o in objs]
    try:
        yield
    finally:
        for o, pv in reversed(list(zip(objs, prev_values))):
            _params_as_tensors_exit(o, pv)


def autoflow(*af_args, **af_kwargs):
    def autoflow_wrapper_decorator(method):
        @functools.wraps(method)
        def autoflow_wrapper(obj, *args, **kwargs):
            if not isinstance(obj, Node):
                raise GPflowError(
                    'AutoFlow works only with node-like objects.')
            if obj.is_built_coherence(obj.graph) is Build.NO:
                raise GPflowError('Not built with "{graph}".'.format(graph=obj.graph))
            name = method.__name__
            store = AutoFlow.get_autoflow(obj, name)
            session = kwargs.pop('session', None)
            session = obj.enquire_session(session=session)

            scope_name = _name_scope_name(obj, name)
            with session.graph.as_default(), tf.name_scope(scope_name):
                if not store:
                    _setup_storage(store, *af_args, **af_kwargs)
                    _build_method(method, obj, store)
                return _session_run(session, obj, store, *args, **kwargs)
        return autoflow_wrapper
    return autoflow_wrapper_decorator


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
    store['arguments'] = [tf.placeholder(*arg) for arg in args]


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
    initialize = kwargs.pop('initialize', False)
    obj.initialize(session=session, force=initialize)
    return session.run(store['result'], **kwargs)


def _build_method(method, obj, store):
    store['result'] = method(obj, *store['arguments'])
