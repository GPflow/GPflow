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

__TRAINABLES = tf.GraphKeys.TRAINABLE_VARIABLES


class GPflowError(Exception):
    pass


def is_tensorflow_variable(value):
    return isinstance(value, (tf.Tensor, tf.Variable))


def is_number(value):
    return not isinstance(value, str) and np.isscalar(value)


def add_to_trainables(variable, graph=None):
    graph = tf.get_default_graph() if graph is None else graph
    if variable not in graph.get_collection(__TRAINABLES):
        graph.add_to_collection(__TRAINABLES, variable)


def remove_from_trainables(variable, graph=None):
    graph = tf.get_default_graph() if graph is None else graph
    trainables = graph.get_collection_ref(__TRAINABLES)
    if variable not in trainables:
        msg = 'TensorFlow variable {variable} not found in the graph {graph}'
        raise ValueError(msg.format(variable=variable, graph=graph))
    trainables.remove(variable)


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
