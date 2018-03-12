import tensorflow as tf

from . import settings
from .decors import params_as_tensors
from .features import InducingPoints, InducingFeature
from .features import Kuu, Kuf
from .kernels import Kernel, Combination
from . import kernels
from .params import Parameter
from .dispatch import dispatch

from .multifeatures import SeparateIndependentMof, SeparateMixedMof, SharedIndependentMof



# TODO MultiOutputKernels have a different method signature for K and Kdiag (they take full_cov_output)
# this needs better documentation - especially as the default there is *True* not False as for full_cov

# TODO what are the dimensions of MultiOutputKernel.K, Kuu(), Kuf()?
# do we need MultiOutputInducingPoints as a special case ?
# or can we write it in terms of the regular InducingPoints?

class Mok(Kernel):
    """
    Multi Output Kernel

    This is the general multi output correlated kernel.
    K(X1, X2): N1 x P x N2 x P
    """
    pass


class SharedIndependentMok(Mok):
    """
    > Shared: we use the same kernel for each latent GP
    > Independent: Latents are uncorrelated a priori.
    """
    def __init__(self, kern: Kernel, name=None):
        Mok.__init__(self, kern.input_dim, name)
        self.kern = kern

    def K(self, X, X2=None):
        return self.kern.K(X, X2)  # N x N2

    def Kdiag(self, X):
        return self.kern.Kdiag(X)  # N 


class SeparateIndependentMok(Mok, Combination):
    """
    > Separate: we use different kernel for each output latent
    > Independent: Latents are uncorrelated a priori.
    """
    def __init__(self, kern_list, name=None):
        Combination.__init__(self, kern_list, name)

    def K(self, X, X2=None):
        return tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # P x N x N2

    def Kdiag(self, X):
        return tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x P


class SeparateMixedMok(Mok, Combination):
    """
    Linear mixing of the latent GPs to form the output
    """

    def __init__(self, kern_list, W, name=None):
        Combination.__init__(self, kern_list, name)
        self.W = Parameter(W)  # P x L

    @params_as_tensors
    def K(self, X, X2=None, full_cov_output=True):
        Kxx = tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # L x N x N2
        KxxW = Kxx[None, :, :, :] * self.W[:, :, None, None]  # P x L x N x N2
        if full_cov_output:
            # return tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)
            WKxxW = tf.tensordot(self.W, KxxW, [[1], [1]])  # P x P x N x N2
            return tf.transpose(WKxxW, [2, 0, 3, 1])  # N x P x N2 x P
        else:
            # return tf.einsum('lnm,kl,kl->knm', Kxx, self.W, self.W)
            return tf.reduce_sum(self.W[:, :, None, None] * KxxW, [1])  # P x N x N2

    @params_as_tensors
    def Kdiag(self, X, full_cov_output=True):
        K = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x L
        if full_cov_output:
            # Can currently not use einsum due to unknown shape from `tf.stack()`
            # return tf.einsum('nl,lk,lq->nkq', K, self.W, self.W)  # N x P x P
            Wt = tf.transpose(self.W)  # L x P
            return tf.reduce_sum(K[:, :, None, None] * Wt[None, :, :, None] * Wt[None, :, None, :], axis=1)  # N x P x P
        else:
            # return tf.einsum('nl,lk,lk->nkq', K, self.W, self.W)  # N x P
            return tf.matmul(K, self.W ** 2.0, transpose_b=True)  # N x L  *  L x P  ->  N x P


##
## ------------------------ Kuf --------------------------
##

@dispatch(SharedIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    return Kuf(feat.feat, kern.kern, Xnew)  # M x N


@dispatch(SeparateIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(f, kern.kern, Xnew) for f in feat.feat_list], axis=0)  # L x M x N


@dispatch(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok), object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kern_list], axis=0)  # L x M x N


@dispatch(SeparateIndependentMof, (SeparateIndependentMok, SeparateMixedMok), object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # L x M x N


@dispatch(SeparateMixedMof, SeparateMixedMok, object)
def Kuf(feat, kern, Xnew):
    Kstack = tf.stack([Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kern_list)], axis=1)  # M x L x N
    return Kstack[:, :, :, None] * tf.transpose(kern.W)[None, :, None, :]  # M x L x N x P


# @dispatch(MultiOutputInducingPoints, Kernel, object)
# def Kuf(feat, kern, Xnew):
#     """
#     TODO: return shape (M*L) x N ?
#     """
#     Kzx = kern.K(feat.Z, Xnew, full_cov_output=True)

# @dispatch()
# def Kuf(feat: InducingFeature, kern: MultiOutputKernel, Xnew: object):
#     return tf.stack([Kuf(feat, k, Xnew) for kern in k.kern_list], axis=0)  # (P or L) x M x N

# @dispatch()
# def Kuf(feat: InducingFeature, kern: MixedMultiOutputKernel, Xnew: object):
#     Kstack = tf.stack([Kuf(feat, k, Xnew) for k in kern.kern_list], axis=1)  # M x L x N
#     return Kstack[:, :, :, None] * tf.transpose(kern.P)[None, :, None, :]


##
## ------------------------ Kuu --------------------------
##


@dispatch(SharedIndependentMof, SharedIndependentMok)
def Kuu(feat, kern, *, jitter=0.0):
    Kmm = Kuu(feat.feat, kern.kern)  # M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type) * jitter
    return Kmm + jittermat


@dispatch(SeparateIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    Kmm = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kern_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(SeparateMixedMof, SeparateMixedMok)
def Kuu(feat, kern, *, jitter=0.0):
    Kxx = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # L x M x M
    WKxxWT = tf.einsum('lnm,kl,ql->nkmq', Kxx, self.W, self.W)  # M x P x M x P
    return WKxxWT


# @dispatch(SeparateIndependentMof, SeparateIndependentMok)
# def Kuu(feat, kern, *, jitter=0.0):
#     """
#     TODO: Return shape (M*L) x (M*L) ?
#     """
#     with params_as_tensors_for(feat, kern):
#         Kzz = kern.K(feat.Z, full_cov_output=True)
#         num_inducing_variables = tf.shape(Kzz)[0] * tf.shape(Kzz)[1]
#         Kzz += jitter * tf.reshape(tf.eye(num_inducing_variables, dtype=settings.dtypes.float_type), tf.shape(Kzz))
#     return Kzz

# @dispatch()
# def Kuu(feat: InducingFeature, kern: IndependentMultiOutputKernel, *, jitter=0.0):
#     Kmm = tf.stack([Kuu(feat, k) for k in kern.kern_list], axis=0)  # (P or L) x M x M
#     jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
#     return Kmm + jittermat

# @dispatch()
# def Kuu(feat: SeparateInducingFeatures, kern: IndependentMultiOutputKernel, *, jitter=0.0):
#     assert(len(feat.feat_list) == len(kern.kern_list))
#     Kmm = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # (P or L) x M x M
#     jittermat = tf.eye(len(feat), dtype=settings.float_type)[None, :, :] * jitter
#     return Kmm + jittermat

# @dispatch()
# def Kuu(feat: SeparateInducingFeatures, kern: MixedMultiOutputKernel, *, jitter=0.0):
#     raise NotImplementedError

