import abc

import tensorflow as tf

from .kernels import Combination
from .params import Parameter


class IMultiKernel(abc.ABCMeta):
    pass


class ListKernel(Combination):
    def K(self, X, X2=None, presliced=False, full_cov_output=False):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        if full_cov_output:
            stacked = tf.stack([k.K(X, X2) for k in self.kern_list], axis=2)  # N x N2 x L
            stacked_diag = tf.matrix_diag(stacked)  # N x N2 x L x L
            return tf.transpose(stacked_diag, [0, 2, 1, 3])  # N x L x N2 x L
        else:
            return tf.stack([k.K(X, X2) for k in self.kern_list])  # L x N x N2

    def Kdiag(self, X, presliced=False, full_cov_output=False):
        stacked = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x L
        return tf.matrix_diag(stacked) if full_cov_output else stacked  # N x L x L or # N x L


# class MixedMultiOutputGP(Combination):
#     def __init__(self, kern_list, P, name):
#         Combination.__init__(self, kern_list, name)
#         self.P = Parameter(P)
#
#     def K(self, X, X2, presliced=False, full_cov_output=False):
