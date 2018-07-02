# Copyright 2016 James Hensman, alexggmatthews
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

import tensorflow as tf
import numpy as np
import pandas as pd
from collections import OrderedDict

from . import settings
from ._version import __version__


__TRAINABLES = tf.GraphKeys.TRAINABLE_VARIABLES
__GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


def pretty_pandas_table(row_names, column_names, column_values):
    return pd.DataFrame(
        OrderedDict(zip(column_names, column_values)),
        index=row_names)


def tensor_name(*subnames):
    return '/'.join(subnames)


def get_variable_by_name(name, graph=None):
    graph = _get_graph(graph)
    return _get_variable(name, graph=graph)


def get_tensor_by_name(name, index=None, graph=None):
    graph = _get_graph(graph)
    return _get_tensor(name, index=index, graph=graph)


def is_ndarray(value):
    return isinstance(value, np.ndarray)


def is_list(value):
    return isinstance(value, list)


def is_tensor(value):
    return isinstance(value, (tf.Tensor, tf.Variable))


def is_number(value):
    return (not isinstance(value, str)) and np.isscalar(value)


def is_valid_param_value(value):
    if isinstance(value, list):
        if not value:
            return False
        zero_val = value[0]
        arrays = (list, np.ndarray)
        scalars = (float, int)
        if isinstance(zero_val, scalars):
            types = scalars
        elif isinstance(zero_val, arrays):
            types = arrays
        else:
            return False
        return all(isinstance(val, types) for val in value[1:])
    return ((value is not None)
            and is_number(value)
            or is_ndarray(value)
            or is_tensor(value))


def is_tensor_trainable(tensor):
    return tensor in tensor.graph.get_collection(__TRAINABLES)


def is_initializable_tensor(tensor):
    return hasattr(tensor, 'initializer')


def add_to_trainables(variable, graph=None):
    graph = _get_graph(graph)
    if variable not in graph.get_collection(__TRAINABLES):
        graph.add_to_collection(__TRAINABLES, variable)


def remove_from_trainables(variable, graph=None):
    graph = _get_graph(graph)
    trainables = graph.get_collection_ref(__TRAINABLES)
    if variable not in trainables:
        msg = 'TensorFlow variable {variable} not found in the graph {graph}'
        raise ValueError(msg.format(variable=variable, graph=graph))
    trainables.remove(variable)


def normalize_num_type(num_type):
    """
    Work out what a sensible type for the array is. if the default type
    is float32, downcast 64bit float to float32. For ints, assume int32
    """
    if isinstance(num_type, tf.DType):
        num_type = num_type.as_numpy_dtype.type

    if num_type in [np.float32, np.float64]:  # pylint: disable=E1101
        num_type = settings.float_type
    elif num_type in [np.int16, np.int32, np.int64]:
        num_type = settings.int_type
    else:
        raise ValueError('Unknown dtype "{0}" passed to normalizer.'.format(num_type))

    return num_type


# def types_array(tensor, shape=None):
#     shape = shape if shape is not None else tensor.shape.as_list()
#     return np.full(shape, tensor.dtype).tolist()


def get_attribute(obj, name, allow_fail=False, default=None):
    try:
        return object.__getattribute__(obj, name)
    except AttributeError as error:
        if allow_fail:
            return default
        raise error


def vec_to_tri(vectors, N):
    """
    Takes a D x M tensor `vectors' and maps it to a D x matrix_size X matrix_sizetensor
    where the where the lower triangle of each matrix_size x matrix_size matrix is
    constructed by unpacking each M-vector.

    Native TensorFlow version of Custom Op by Mark van der Wilk.

    def int_shape(x):
        return list(map(int, x.get_shape()))

    D, M = int_shape(vectors)
    N = int( np.floor( 0.5 * np.sqrt( M * 8. + 1. ) - 0.5 ) )
    # Check M is a valid triangle number
    assert((matrix * (N + 1)) == (2 * M))
    """
    indices = list(zip(*np.tril_indices(N)))
    indices = tf.constant([list(i) for i in indices], dtype=tf.int64)

    def vec_to_tri_vector(vector):
        return tf.scatter_nd(indices=indices, shape=[N, N], updates=vector)

    return tf.map_fn(vec_to_tri_vector, vectors)


def initialize_variables(variables=None, session=None, force=False, **run_kwargs):
    session = tf.get_default_session() if session is None else session
    if variables is None:
        initializer = tf.global_variables_initializer()
    else:
        if force:
            vars_for_init = list(_initializable_tensors(variables))
        else:
            vars_for_init = list(_find_initializable_tensors(variables, session))
        if not vars_for_init:
            return
        initializer = tf.variables_initializer(vars_for_init)
    session.run(initializer, **run_kwargs)


def _initializable_tensors(initializables):
    for v in initializables:
        if isinstance(v, (tuple, list)):
            yield v[0]
        else:
            yield v


def _find_initializable_tensors(intializables, session):
    for_reports = []
    status_tensors = []
    boolean_tensors = []

    for v in intializables:
        if isinstance(v, (tuple, list)):
            status_tensors.append(v[0])
            boolean_tensors.append(v[1])
        # TODO(@awav): Tensorflow Iterator must have to be skipped at
        # auto-intialization unless TensorFlow issue #14633 is resolved.
        elif isinstance(v, tf.data.Iterator):
            continue
        else:
            for_reports.append(v)

    if for_reports:
        uninitialized = tf.report_uninitialized_variables(var_list=for_reports)
        def uninitialized_names():
            for uv in session.run(uninitialized):
                yield uv.decode('utf-8')

        names = set(uninitialized_names())
        for v in for_reports:
            if v.name.split(':')[0] in names:
                yield v

    if boolean_tensors:
        stats = session.run(boolean_tensors)
        length = len(stats)
        for i in range(length):
            if not stats[i]:
                yield status_tensors[i]


def _get_graph(graph=None):
    return tf.get_default_graph() if graph is None else graph


def _get_tensor(name, index=None, graph=None):
    graph = _get_graph(graph)
    if index is not None:
        return _get_tensor_safe(name, index, graph)
    tensor = _get_tensor_safe(name, '0', graph)
    if tensor is None:
        return tensor
    if _get_tensor_safe(name, '1', graph) is not None:
        raise ValueError('Ambiguous tensor for "{0}" with multiple indices found.'
                         .format(name))
    return tensor


def _get_variable(name, graph=None):
    for var in graph.get_collection(__GLOBAL_VARIABLES):
        var_name, _var_index = var.name.split(':')
        if var_name == name:
            return var
    return None


def _get_tensor_safe(name, index, graph):
    try:
        return graph.get_tensor_by_name(':'.join([name, index]))
    except KeyError:
        return None

def version():
    return __version__
