# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

"""
Throughout GPflow, by default, latent functions being modelled with Gaussian
processes are assumed to have zero mean, f ~ GP(0, k(x,x')).

In some cases we may wish to model only the deviation from a fixed function
with a Gaussian process.  For flexibility this fixed function could be both
input dependent and parameterised function, μ(x; θ),
with some unknown parameters θ, resulting in f ~ GP(μ(x;θ), k(x,x')).

The GPflow :class:`MeanFunction <gpflow.mean_functions.MeanFunction>` class
allows this to be done whilst additionally learning parameters of the
parametric function.
"""

import numpy as np
import tensorflow as tf

from .base import Module, Parameter
from .config import default_float, default_int


class MeanFunction(Module):
    """
    The base mean function class.
    To implement a mean function, write the __call__ method. This takes a
    tensor X and returns a tensor m(X). In accordance with the GPflow
    standard, each row of X represents one datum, and each row of Y is computed
    independently for each row of X.

    MeanFunction classes can have parameters, see the Linear class for an
    example.
    """

    def __call__(self, X):
        raise NotImplementedError("Implement the __call__ method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """

    def __init__(self, A=None, b=None):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be [D, Q], b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        A = np.ones((1, 1), dtype=default_float()) if A is None else A
        b = np.zeros(1, dtype=default_float()) if b is None else b
        self.A = Parameter(np.atleast_2d(A))
        self.b = Parameter(b)

    def __call__(self, X):
        return tf.tensordot(X, self.A, [[-1], [0]]) + self.b


class Identity(Linear):
    """
    y_i = x_i
    """

    def __init__(self, input_dim=None):
        Linear.__init__(self)
        self.input_dim = input_dim

    def __call__(self, X):
        return X

    @property
    def A(self):
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )
        return tf.eye(self.input_dim, dtype=default_float())

    @property
    def b(self):
        if self.input_dim is None:
            raise ValueError(
                "An input_dim needs to be specified when using the "
                "`Identity` mean function in combination with expectations."
            )

        return tf.zeros(self.input_dim, dtype=default_float())

    @A.setter
    def A(self, A):
        pass

    @b.setter
    def b(self, b):
        pass


class Constant(MeanFunction):
    def __init__(self, c=None):
        super().__init__()
        c = np.zeros(1) if c is None else c
        self.c = Parameter(c)

    def __call__(self, X):
        tile_shape = tf.concat([tf.shape(X)[:-1], [1]], axis=0,)
        reshape_shape = tf.concat(
            [tf.ones(shape=(tf.rank(X) - 1), dtype=default_int()), [-1]], axis=0,
        )
        return tf.tile(tf.reshape(self.c, reshape_shape), tile_shape)


class Zero(Constant):
    def __init__(self, output_dim=1):
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def __call__(self, X):
        output_shape = tf.concat([tf.shape(X)[:-1], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=X.dtype)


class SwitchedMeanFunction(MeanFunction):
    """
    This class enables to use different (independent) mean_functions respective
    to the data 'label'.
    We assume the 'label' is stored in the extra column of X.
    """

    def __init__(self, meanfunction_list):
        super().__init__()
        for m in meanfunction_list:
            assert isinstance(m, MeanFunction)
        self.meanfunctions = meanfunction_list

    def __call__(self, X):
        ind = tf.gather(tf.transpose(X), tf.shape(X)[1] - 1)  # ind = X[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(
            tf.gather(tf.transpose(X), tf.range(0, tf.shape(X)[1] - 1))
        )  # X = X[:,:-1]

        # split up X into chunks corresponding to the relevant likelihoods
        x_list = tf.dynamic_partition(X, ind, len(self.meanfunctions))
        # apply the likelihood-function to each section of the data
        results = [m(x) for x, m in zip(x_list, self.meanfunctions)]
        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.meanfunctions))
        return tf.dynamic_stitch(partitions, results)


class Additive(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)
        self.add_1 = first_part
        self.add_2 = second_part

    def __call__(self, X):
        return tf.add(self.add_1(X), self.add_2(X))


class Product(MeanFunction):
    def __init__(self, first_part, second_part):
        MeanFunction.__init__(self)

        self.prod_1 = first_part
        self.prod_2 = second_part

    def __call__(self, X):
        return tf.multiply(self.prod_1(X), self.prod_2(X))
