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

from . import settings
from .core.errors import GPflowError


__TRAINABLES = tf.GraphKeys.TRAINABLE_VARIABLES
__GLOBAL_VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES


def tensor_name(*subnames):
    return '/'.join(subnames)


def get_variable_by_name(name, index=None, graph=None):
    graph = _get_graph(graph)
    return _get_variable(name, index=index, graph=graph)


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
        zero_type = type(value[0])
        return all(isinstance(val, zero_type) for val in value[1:])
    return ((value is not None)
            and is_number(value)
            or is_ndarray(value)
            or is_tensor(value))


def initialize_variables(variables=None, session=None, force=False, **run_kwargs):
    session = tf.get_default_session() if session is None else session
    if variables is None:
        initializer = tf.global_variables_initializer()
    else:
        if force:
            initializer = tf.variables_initializer(variables)
        else:
            uninitialized = tf.report_uninitialized_variables(var_list=variables)
            def uninitialized_names():
                for uv in session.run(uninitialized):
                    if isinstance(uv, bytes):
                        yield uv.decode('utf-8')
                    elif isinstance(uv, str):
                        yield uv
                    else:
                        msg = 'Unknown output type "{}"from `tf.report_uninitialized_variables`'
                        raise ValueError(msg.format(type(uv)))
            names = set(uninitialized_names())
            vars_for_init = [v for v in variables if v.name.split(':')[0] in names]
            initializer = tf.variables_initializer(vars_for_init)
    session.run(initializer, **run_kwargs)


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
        raise GPflowError(msg.format(variable=variable, graph=graph))
    trainables.remove(variable)


def normalize_num_type(num_type):
    """
    Work out what a sensible type for the array is. if the default type
    is float32, downcast 64bit float to float32. For ints, assume int32
    """
    if isinstance(num_type, tf.DType):
        num_type = num_type.as_numpy_dtype.type

    if num_type in [np.float32, np.float64]:  # pylint: disable=E1101
        num_type = settings.np_float
    elif num_type in [np.int16, np.int32, np.int64]:
        num_type = settings.np_int
    else:
        raise ValueError('Unknown dtype "{0}" passed to normalizer.'.format(num_type))

    return num_type


def types_array(tensor, shape=None):
    shape = shape if shape is not None else tensor.shape.as_list()
    return np.full(shape, tensor.dtype).tolist()


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


def _get_variable(name, index=None, graph=None):
    variables = []
    for var in graph.get_collection(__GLOBAL_VARIABLES):
        var_name, var_index = var.name.split(':')
        if var_name == name:
            if index is not None and var_index == index:
                return var
            variables.append(var)
    if index is not None or not variables:
        return None
    if len(variables) > 1:
        raise ValueError('Ambiguous variable for "{0}" with multiple indices found.')
    return variables[0]


def _get_tensor_safe(name, index, graph):
    try:
        return graph.get_tensor_by_name(':'.join([name, index]))
    except KeyError:
        return None
