import tensorflow as tf

from . import settings
from .decors import params_as_tensors
from .features import InducingPoints, InducingFeature
from .kernels import Kernel, Combination
from .params import Parameter

# TODO MultiOutputKernels have a different method signature for K and Kdiag (they take full_cov_output)
# this needs better documentation - especially as the default there is *True* not False as for full_cov

class MultiOutputKernel(Kernel):
    pass


class IndependentFeature(InducingFeature):
    pass


class MultiOutputInducingPoints(IInducingPoints):
    """
    TODO
    """

@dispatch(MultiOutputInducingPoints, kernels.Kernel)
def Kuu(feat, kern, jitter=0.0):
    with params_as_tensors_for(feat, kern):
        Kzz = kern.K(feat.Z, full_cov_output=True)
        num_inducing_variables = tf.shape(Kzz)[0] * tf.shape(Kzz)[1]
        Kzz += jitter * tf.reshape(tf.eye(num_inducing_variables, dtype=settings.dtypes.float_type), tf.shape(Kzz))
    return Kzz

@dispatch(MultiOutputInducingPoints, kernels.Kernel, object)
def Kuf(feat, kern, Xnew):
    with params_as_tensors_for(feat, kern):
        Kzx = kern.K(feat.Z, Xnew, full_cov_output=True)
    return Kzx


class IndependentMultiOutputKernel(Combination, MultiOutputKernel):
    """
    TODO
    """

    def K(self, X, X2=None, presliced=False, full_cov_output=True):
        if presliced:
            # Haven't thought about what to do here
            # TODO presliced seems to only be relevant for evaluating the
            # individual kernels in a sum / product, so maybe we can just
            # ignore it here, just as Sum / Product kernels?
            raise NotImplementedError()
        if full_cov_output:
            stacked = tf.stack([k.K(X, X2) for k in self.kern_list], axis=2)  # N x N2 x P
            stacked_diag = tf.matrix_diag(stacked)  # N x N2 x P x P
            return tf.transpose(stacked_diag, [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # P x N x N2

    def Kdiag(self, X, presliced=False, full_cov_output=False):
        if presliced:
            raise NotImplementedError
        stacked = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x P
        return tf.matrix_diag(stacked) if full_cov_output else stacked  # N x P x P  or  N x P


class IndependentSharedInducingPoints(InducingPoints, IndependentFeature):
    """
    TODO
    """

@dispatch
def Kuf(feat: IndependentSharedInducingPoints, kern: IndependentMultiOutputKernel, Xnew: object):
    # TODO should this not just call Kuf() for the base cases?
    return tf.stack([Kuf(feat, k, Xnew) for k in kern.kern_list], axis=0)  # P x M x N

@dispatch
def Kuf(feat: IndependentSharedInducingPoints, kern: MixedMultiOutputKernel, Xnew: object):
    Kstack = tf.stack([Kuf(feat, k, Xnew) for k in kern.kern_list], axis=1)  # M x L x N
    return Kstack[:, :, :, None] * tf.transpose(kern.P)[None, :, None, :]

@dispatch
def Kuu(feat: IndependentSharedInducingPoints, kern: MultiOutputKernel, jitter=0.0):
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return tf.stack([Kuu(feat, k) for k in kern.kern_list], axis=0) + jittermat  # P x M x M


class MixedMultiIndependentFeature(InducingPoints):  # TODO better name?
    """
    TODO
    """

@dispatch
def Kuf(feat, kern, Xnew):
    # TODO how is this different from the case above?
    return tf.stack([Kuf(feat, kern, Xnew) for kern in kern.kern_list], axis=0)  # L x M x N

def Kuu(self, kern, jitter=0.0):
    # TODO how is this different from the case above?
    Kmm = tf.stack([Kuu(feat, k) for k in kern.kern_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(self), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat

class MixedMultiOutputKernel(Combination, MultiOutputKernel):
    """
    TODO
    """

    def __init__(self, kern_list, P, name=None):
        super().__init__(kern_list, name)
        # TODO need to check shape of P
        self.P = Parameter(P)  # P x L

    @params_as_tensors
    def K(self, X, X2=None, presliced=False, full_cov_output=True):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        Kxx = tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # L x N x N2
        KxxP = Kxx[None, :, :, :] * self.P[:, :, None, None]  # P x L x N x N2
        if full_cov_output:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.P, self.P)
            PKxxP = tf.tensordot(self.P, KxxP, [[1], [1]])  # K x K x N x N2
            return tf.transpose(PKxxP, [2, 0, 3, 1])  # N x K x N2 x K
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.P, self.P)
            return tf.reduce_sum(self.P[:, :, None, None] * KxxP, [1])  # K x N x N2

    @params_as_tensors
    def Kdiag(self, X, presliced=False, full_cov_output=True):
        if presliced:
            # Haven't thought about what to do here
            raise NotImplementedError()
        K = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x L
        if full_cov_output:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.P, self.P)  # N x P x P
            Pt = tf.transpose(self.P)  # L x P
            return tf.reduce_sum(K[:, :, None, None] * Pt[None, :, :, None] * Pt[None, :, None, :], axis=1)  # N x P x P
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.P, self.P)  # N x P
            return tf.matmul(K, self.P ** 2.0, transpose_b=True)  # N x L  *  L x P  ->  N x P


