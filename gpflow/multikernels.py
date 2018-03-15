import tensorflow as tf

from . import settings
from .decors import params_as_tensors, autoflow
from .features import InducingPoints, InducingFeature
from .features import Kuu, Kuf
from .kernels import Kernel, Combination
from . import kernels
from .params import Parameter
from .dispatch import dispatch

from .multifeatures import SeparateIndependentMof, SharedIndependentMof, MixedKernelSharedMof

float_type = settings.float_type

# TODO MultiOutputKernels have a different method signature for K and Kdiag (they take full_cov_output)
# this needs better documentation - especially as the default there is *True* not False as for full_cov

# TODO what are the dimensions of MultiOutputKernel.K, Kuu(), Kuf()?
# do we need MultiOutputInducingPoints as a special case ?
# or can we write it in terms of the regular InducingPoints?

class Mok(Kernel):
    """
    Multi Output Kernel class.

    Kernels parameterize priors over scalar valued functions
    Multi Output Kernel parameterize priors over vector valued functions (dim=P).

    A Kernel parameterizes covariance Cov[f(x),f(x')]
    A MOK parameterizes covariances Cov[f_i(x),f_j(x')] for all pairs of i,k in [1..P]^2,
    that is covariance both across inputs and across output dimensions


    Subclasses of Mok should implement K which returns:
     - N x P x N x P if full_cov_output = True
     - N x N x P if full_cov_output = False

    This is the general multi output correlated kernel.
    K(X1, X2): N1 x P x N2 x P
    """
    pass


class SharedIndependentMok(Mok):
    """
    Note: this class is created only for testing purposes.
    Use `gpflow.kernels` instead for more efficient code.

    > Shared: we use the same kernel for each GP
    > Independent: GPs are uncorrelated a priori.
    """
    def __init__(self, kern: Kernel, output_dimensionality, name=None):
        Mok.__init__(self, kern.input_dim, name)
        self.kern = kern
        self.P = output_dimensionality

    def K(self, X, X2=None, full_cov_output=True):
        K = self.kern.K(X, X2)  # N x N2
        if full_cov_output:
            Ks = tf.tile(K[..., None], [1, 1, self.P])  # N x N2 x P
            return tf.transpose(tf.matrix_diag(Ks), [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.tile(K[None, ...], [self.P, 1, 1])  # P x N x N2
        
    def Kdiag(self, X, full_cov_output=True):
        K = self.kern.Kdiag(X)  # N 
        Ks = tf.tile(K[:, None], [1, self.P])  # N x P
        return tf.matrix_diag(Ks) if full_cov_output else Ks  # N x P x P or N x P


class SeparateIndependentMok(Mok, Combination):
    """
    > Separate: we use different kernel for each GP
    > Independent: GPs are uncorrelated a priori.
    """
    def __init__(self, kern_list, name=None):
        Combination.__init__(self, kern_list, name)

    def K(self, X, X2=None, full_cov_output=True):
        if full_cov_output:
            Kxxs = tf.stack([k.K(X, X2) for k in self.kern_list], axis=2)  # N x N2 x P
            return tf.transpose(tf.matrix_diag(Kxxs), [0, 2, 1, 3])  # N x P x N2 x P
        else:
            return tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # P x N x N2

    def Kdiag(self, X, full_cov_output=False):
        stacked = tf.stack([k.Kdiag(X) for k in self.kern_list], axis=1)  # N x P
        return tf.matrix_diag(stacked) if full_cov_output else stacked  # N x P x P  or  N x P


class SeparateMixedMok(Mok, Combination):
    """
    Linear mixing of the latent GPs (g, dim=L) to form correlated output (f, dim=P)
    """

    def __init__(self, kern_list, W, name=None):
        Combination.__init__(self, kern_list, name)
        self.W = Parameter(W)  # P x L

    @params_as_tensors
    def Kgg(self, X, X2):
        return tf.stack([k.K(X, X2) for k in self.kern_list], axis=0)  # L x N x N2
    
    @autoflow((settings.float_type, [None, None]),
              (settings.float_type, [None, None]))
    def compute_Kgg(self, X, X2):
        return self.Kgg(X, X2)

    @params_as_tensors
    def K(self, X, X2=None, full_cov_output=True):
        Kxx = self.Kgg(X, X2)  # L x N x N2
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


# ============================= Kuf =============================

@dispatch(InducingPoints, Mok, object)
def Kuf(feat, kern, Xnew):
    print("Kuf: InducingPoints - Mok")
    return kern.K(feat.Z, Xnew, full_cov_output=True)  #  M x P x N x P


@dispatch(SharedIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    print("Kuf: SharedIndependentMof - SharedIndependentMok")
    return Kuf(feat.feat, kern.kern, Xnew)  # M x N


@dispatch(SeparateIndependentMof, SharedIndependentMok, object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(f, kern.kern, Xnew) for f in feat.feat_list], axis=0)  # L x M x N


@dispatch(SharedIndependentMof, SeparateIndependentMok, object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kern_list], axis=0)  # L x M x N


@dispatch(SeparateIndependentMof, SeparateIndependentMok, object)
def Kuf(feat, kern, Xnew):
    return tf.stack([Kuf(f, k, Xnew) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # L x M x N


@dispatch((SeparateIndependentMof, SharedIndependentMof), SeparateMixedMok, object) 
def Kuf(feat, kern, Xnew):
    kuf_impl = Kuf.dispatch(type(feat), SeparateIndependentMok, object)
    K = tf.transpose(kuf_impl(feat, kern, Xnew), [1, 0, 2])  # M x L x N
    return K[:, :, :, None] * tf.transpose(kern.W)[None, :, None, :]  # M x L x N x P


@dispatch(MixedKernelSharedMof, SeparateMixedMok, object)
def Kuf(feat, kern, Xnew):
    print("Kuf: MixedKernelSharedMof, SeparateMixedMok")
    return tf.stack([Kuf(feat.feat, k, Xnew) for k in kern.kern_list], axis=0)  # L x M x N


# ============================= Kuu =============================

@dispatch(InducingPoints, Mok)
def Kuu(feat, kern, *, jitter=0.0):
    print("Kuu: InducingPoints - Mok")
    Kmm = kern.K(feat.Z, full_cov_output=True)  # M x P x M x P
    M = tf.shape(Kmm)[0] * tf.shape(Kmm)[1]
    jittermat = jitter * tf.reshape(tf.eye(M, dtype=float_type), tf.shape(Kmm))
    return Kmm + jittermat


@dispatch(SharedIndependentMof, SharedIndependentMok)
def Kuu(feat, kern, *, jitter=0.0):
    print("Kuu: SharedIndependentMof - SharedIndependentMok")
    Kmm = Kuu(feat.feat, kern.kern)  # M x M
    jittermat = tf.eye(len(feat), dtype=float_type) * jitter
    return Kmm + jittermat


@dispatch(SharedIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kern_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(SeparateIndependentMof, (SeparateIndependentMok, SeparateMixedMok))
def Kuu(feat, kern, *, jitter=0.0):
    Kmm = tf.stack([Kuu(f, k) for f, k in zip(feat.feat_list, kern.kern_list)], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=float_type)[None, :, :] * jitter
    return Kmm + jittermat


@dispatch(MixedKernelSharedMof, SeparateMixedMok)
def Kuu(feat, kern, *, jitter=0.0):
    print("Kuu: MixedKernelSharedMof, SeparateMixedMok")
    Kmm = tf.stack([Kuu(feat.feat, k) for k in kern.kern_list], axis=0)  # L x M x M
    jittermat = tf.eye(len(feat), dtype=float_type)[None, :, :] * jitter
    return Kmm + jittermat