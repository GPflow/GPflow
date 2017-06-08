# Copyright 2016 James Hensman, Mark van der Wilk, Valentine Svensson, alexggmatthews, PabloLeon, fujiisoup
# Copyright 2017 ST John
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


from __future__ import absolute_import
import types
from functools import wraps
import tensorflow as tf
from ._settings import settings
from . import session
float_type = settings.dtypes.float_type


class AutoFlow:
    """
    This decorator-class is designed for use on methods of the Parameterized class
    (below).

    The idea is that methods that compute relevant quantities (such as
    predictions) can define a tf graph which we automatically run when the
    decorated function is called.

    The syntax looks like:

    >>> class MyClass(Parameterized):
    >>>
    >>>   @AutoFlow((tf.float64), (tf.float64))
    >>>   def my_method(self, x, y):
    >>>       #compute something, returning a tf graph.
    >>>       return tf.foo(self.baz, x, y)

    >>> m = MyClass()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> result = my_method(x, y)

    Now the output of the method call is the _result_ of the computation,
    equivalent to

    >>> m = MyModel()
    >>> x = np.random.randn(3,1)
    >>> y = np.random.randn(3,1)
    >>> x_tf = tf.placeholder(tf.float64)
    >>> y_tf = tf.placeholder(tf.float64)
    >>> with m.tf_mode():
    >>>     graph = tf.foo(m.baz, x_tf, y_tf)
    >>> result = m._session.run(graph,
                                feed_dict={x_tf:x,
                                           y_tf:y,
                                           m._free_vars:m.get_free_state()})

    Not only is the syntax cleaner, but multiple calls to the method will
    result in the graph being constructed only once.

    AutoFlow takes as many arguments as the wrapped function, each being a
    tuple with the same arguments as you would give to tf.placeholder. For
    example, (tf.float64, [None,2,3]) indicates a float64 tensor of rank 3,
    with an arbitrary number of elements along the first dimension, 2 elements
    along the second dimension, and 3 elements along the third dimension.
    """

    def __init__(self, *tf_arg_tuples):
        # NB. TF arg_tuples is a list of tuples, each of which can be used to
        # construct a tf placeholder.
        self.tf_arg_tuples = tf_arg_tuples

    def __call__(self, tf_method):
        @wraps(tf_method)
        def runnable(instance, *np_args):
            storage_name = '_' + tf_method.__name__ + '_AF_storage'
            if hasattr(instance, storage_name):
                # the method has been compiled already, get things out of storage
                storage = getattr(instance, storage_name)
            else:
                # the method needs to be compiled.
                storage = {}  # an empty dict to keep things in
                setattr(instance, storage_name, storage)
                storage['graph'] = tf.Graph()
                # storage['session'] = tf.Session(graph=storage['graph'])
                storage['session'] = session.get_session(
                    graph=storage['graph'],
                    output_file_name=settings.profiling.output_file_name + "_" + tf_method.__name__,
                    output_directory=settings.profiling.output_directory,
                    each_time=settings.profiling.each_time
                )
                with storage['graph'].as_default():
                    storage['tf_args'] = [tf.placeholder(*a) for a in self.tf_arg_tuples]
                    storage['free_vars'] = tf.placeholder(float_type, [None])
                    instance.make_tf_array(storage['free_vars'])
                    with instance.tf_mode():
                        storage['tf_result'] = tf_method(instance, *storage['tf_args'])
                    storage['feed_dict_keys'] = instance.get_feed_dict_keys()
                    feed_dict = {}
                    instance.update_feed_dict(storage['feed_dict_keys'], feed_dict)
                    storage['session'].run(tf.global_variables_initializer(), feed_dict=feed_dict)
            feed_dict = dict(zip(storage['tf_args'], np_args))
            feed_dict[storage['free_vars']] = instance.get_free_state()
            instance.update_feed_dict(storage['feed_dict_keys'], feed_dict)
            return storage['session'].run(storage['tf_result'], feed_dict=feed_dict)

        return runnable


def AutoFlowify(*args):
    """
    Decorator that saves its arguments in the `_autoflow_args` field in the
    function so that they can be passed to AutoFlow when constructing a
    compute_method in `make_compute_method`.
    """
    def decorator(func):
        func._autoflow_args = args
        return func
    return decorator


def make_compute_method(compute_name, build_method, autoflow_args):
    """
    Returns a new method with name `compute_name` that calls `build_method`
    but is wrapped by the AutoFlow decorator with the arguments in
    `autoflow_args`.

    `compute_name` should correspond to the name given to the new method in
    the model class.
    """
    @wraps(build_method)
    def compute_method(self, *build_args):
        return build_method(self, *build_args)

    # AutoFlow uses <method>.__name__ for its graph-caching, so we need to
    # keep the original method name, and we need to apply the AutoFlow
    # decorator *after* changing the method's __name__ attribute
    compute_method.__name__ = compute_name
    compute_method = AutoFlow(*autoflow_args)(compute_method)

    return compute_method


class MetaAutoFlow(type):
    """
    Metaclass for GPflow models that automatically constructs an @AutoFlow-
    wrapped compute_<method> for each build_<method> that is decorated with
    @AutoFlowify.

    E.g., when adding this as a metaclass to a model,

    >>> @AutoFlowify((float_type, [None,None]))
    >>> def build_foo(self, a):
    >>>     return tf.reduce_sum(a)

    is equivalent to the more verbose

    >>> @AutoFlow((float_type, [None,None]))
    >>> def compute_foo(self, a):
    >>>     return self.build_foo(a)
    >>>
    >>> def build_foo(self, a):
    >>>     return tf.reduce_sum(a)

    Methods that do not start with 'build_' or that are not decorated by
    @AutoFlowify will be left alone.  If the class defines a compute_ method
    itself, that will take precedence over the metaclass-autogenerated one.
    """

    def __new__(meta, name, bases, class_dict):
        new_class_dict = {}
        for key, value in class_dict.items():
            if (key.startswith('build_') and
                    isinstance(value, types.FunctionType) and
                    hasattr(value, '_autoflow_args')):
                build_name, build_method = key, value

                compute_name = build_name.replace('build_', 'compute_', 1)
                compute_method = make_compute_method(compute_name, build_method, build_method._autoflow_args)
                new_class_dict[compute_name] = compute_method

        new_class_dict.update(class_dict)  # add all other methods, overwriting compute_<> if already exists

        cls = type.__new__(meta, name, bases, new_class_dict)
        return cls
