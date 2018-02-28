import tensorflow as tf

from .decors import params_as_tensors
from .features import InducingPoints, InducingFeature
from .kernels import Kernel, Combination
from .params import Parameter


class IndependentMultiKernel(Kernel):
    pass


class MultiKernel(Kernel):
    pass


class IndependentFeature(InducingFeature):
    pass


class Independent(Combination, IndependentMultiKernel):
    def K(self, X, X2=None, presliced=False, full_cov_output=False):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        if full_cov_output:
            stacked = tf.stack([k.K(X, X2) for k in self.kern_list], axis=2)  # N x N2 x P
            stacked_diag = tf.matrix_diag(stacked)  # N x N2 x P x P
            return tf.transpose(stacked_diag, [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # P x N x N2

    def Kdiag(self, X, presliced=False, full_cov_output=False):
        stacked = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x P
        return tf.matrix_diag(stacked) if full_cov_output else stacked  # N x P x P  or  N x P


class IndependentFeature(InducingPoints):
    def Kuf(self, kern, Xnew):
        pass  # N x P x M

    def Kuu(self, kern, jitter=0.0):
        pass  # P x M x M


class MixedMulti(MultiKernel):
    def __init__(self, kern_list, P, name=None):
        super().__init__(kern_list, name)
        self.P = Parameter(P)  # P x L

    @params_as_tensors
    def K(self, X, X2=None, presliced=False, full_cov_output=False):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        Kxx = tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # P x N x N2
        KxxP = Kxx[None, :, :, :] * self.P[:, :, None, None]  # K x L x N x N2
        if full_cov_output:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.P, self.P)
            PKxxP = tf.tensordot(self.P, KxxP, [[1], [1]])  # K x K x N x N2
            return tf.transpose(PKxxP, [2, 0, 3, 1])  # N x K x N2 x K
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.P, self.P)
            return tf.reduce_sum(self.P[:, :, None, None] * KxxP, [1])  # K x N x N2

    @params_as_tensors
    def Kdiag(self, X, presliced=False, full_cov_output=False):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        K = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x P
        if full_cov_output:
            return tf.einsum('nl,lk,lq->nkq', K, self.P, self.P)  # N x K x K
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.P, self.P)  # N x K
            return tf.matmul(K, self.P ** 2.0)


class MixedMultiIndependentFeature(InducingPoints):
    def Kuf(self, kern, Xnew):
        pass  # N x P x M x L

    def Kuu(self, kern, jitter=0.0):
        pass  # L x M x M
