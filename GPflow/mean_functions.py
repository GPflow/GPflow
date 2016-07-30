import tensorflow as tf
import numpy as np
from .param import Param, Parameterized


class MeanFunction(Parameterized):
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
        raise NotImplementedError("Implement the __call__\
                                  method for this mean function")

    def __add__(self, other):
        return Additive(self, other)

    def __mul__(self, other):
        return Product(self, other)


class Zero(MeanFunction):
    def __call__(self, X):
        return tf.zeros(tf.pack([tf.shape(X)[0], 1]), dtype='float64')


class Linear(MeanFunction):
    """
    y_i = A x_i + b
    """
    def __init__(self, A=np.ones((1, 1)), b=np.zeros(1)):
        """
        A is a matrix which maps each element of X to Y, b is an additive
        constant.

        If X has N rows and D columns, and Y is intended to have Q columns,
        then A must be D x Q, b must be a vector of length Q.
        """
        MeanFunction.__init__(self)
        self.A = Param(np.atleast_2d(A))
        self.b = Param(b)

    def __call__(self, X):
        return tf.matmul(X, self.A) + self.b


class Constant(MeanFunction):
    """
    y_i = c,,
    """
    def __init__(self, c=np.zeros(1)):
        MeanFunction.__init__(self)
        self.c = Param(c)

    def __call__(self, X):
        shape = tf.pack([tf.shape(X)[0], 1])
        return tf.tile(tf.reshape(self.c, (1, -1)), shape)


class SwitchedMeanFunction(MeanFunction):
    def init__(self, meanfunction_list, input_dim, active_dim):
    """
    In this mean function, we assume at active_dim column of X, wchih contains
    integers that specify a mean_function from the list of mean_functions.
    """
        MeanFunction.__init__(self)
        for m in meanfunction_list:
            assert isinstance(m, meanfunction_list)
        self.meanfunction_list = ParamList(meanfunction_list)
        self.num_meanfunctions = len(self.meanfunction_list)
        # dimension that contains integers
        self.active_dim = active_dim
        # the rest of dimension as list, for deriving the true expressive variables
        self.other_dims = list(range(input_dim))
        pop(self.other_dims[self.active_dim])

    def __call__(self, X):
        ind = tf.gather(tf.transpose(X), self.active_dim)  # ind = Y[:,-1]
        ind = tf.cast(ind, tf.int32)
        X = tf.transpose(tf.gather(tf.transpose(X), self.other_dims))
        # apply the likelihood-function to each section of the data
        results = [m(X) for m in self.meanfunction_list]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_meanfunctions)
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
        return tf.mul(self.prod_1(X), self.prod_2(X))
