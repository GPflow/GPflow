import types

import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

from . import kernels, mean_functions, GPflowError, settings
from .features import InducingFeature, InducingPoints
from .quadrature import mvnquad
from .decors import params_as_tensors_for
from .params import Parameterized, Parameter


class ProbabilityDistribution(Parameterized):
    def __init__(self, name=None):
        super(ProbabilityDistribution, self).__init__(name=name)


class Gaussian(ProbabilityDistribution):
    def __init__(self, mu, cov, name=None):
        super(Gaussian, self).__init__(name=name)
        self.mu = Parameter(mu)  # N x D
        self.cov = Parameter(cov)  # N x D x D

## Generic case, quadrature method

def get_eval_func(obj, feature, slice=np.s_[...]):
    if feature is not None:
        # kernel + feature combination
        if not isinstance(feature, InducingFeature) or not isinstance(obj, kernels.Kern):
            raise GPflowError("If `feature` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(feature.Kuf(obj, x))[slice]
    elif isinstance(obj, mean_functions.MeanFunction):
        return lambda x: obj(x)[slice]
    elif isinstance(obj, kernels.Kern):
        return lambda x: obj.Kdiag(x)
    elif isinstance(obj, types.FunctionType):
        return obj
    else:
        raise NotImplementedError()

def quad_expectation(obj1, feature1, obj2, feature2, p):
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature1)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    with params_as_tensors_for(p):
        return mvnquad(eval_func, p.mu, p.cov, 150)

@dispatch(object, (InducingFeature, type(None)), object, (InducingFeature, type(None)), Gaussian)
def expectation(obj1, feature1, obj2, feature2, p):
    print("MD Quad")
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature1)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    with params_as_tensors_for(p):
        return mvnquad(eval_func, p.mu, p.cov, 20)


# RBF Psi0
@dispatch(kernels.RBF, type(None), type(None), type(None), Gaussian)
def expectation(kern, none1, none2, none3, p):
    """
    Also known as phi_0: <kern{x, x}>_{p(x)}.
    :return: N
    """
    print("MD RBF Psi0")
    with params_as_tensors_for(p):
        return kern.Kdiag(p.mu)

# RBF Psi1
@dispatch(kernels.RBF, InducingPoints, type(None), type(None), Gaussian)
def expectation(kern, feat, none1, none2, p):
    """
    Also known as phi_1: <kern{x, feat.Z}>_{p(x)}.
    :paran kern: RBF
    :param feat: InducingPoints
    :param p: Gaussian
    :return: NxM, with N = shape(p.mu)[0] and M = shape(feat.Z)[0]
    """
    print("MD RBF Psi1")

    with params_as_tensors_for(feat), \
         params_as_tensors_for(kern), \
         params_as_tensors_for(p):

        # use only active dimensions
        Xcov = kern._slice_cov(p.cov)
        Z, Xmu = kern._slice(feat.Z, p.mu)
        D = tf.shape(Xmu)[1]
        if kern.ARD:
            lengthscales = kern.lengthscales
        else:
            lengthscales = tf.zeros((D,), dtype=settings.tf_float) + kern.lengthscales

        vec = tf.expand_dims(Xmu, 2) - tf.expand_dims(tf.transpose(Z), 0)  # NxDxM
        chols = tf.cholesky(tf.expand_dims(tf.matrix_diag(lengthscales ** 2), 0) + Xcov)
        Lvec = tf.matrix_triangular_solve(chols, vec)
        q = tf.reduce_sum(Lvec ** 2, [1])

        chol_diags = tf.matrix_diag_part(chols)  # N x D
        half_log_dets = tf.reduce_sum(tf.log(chol_diags), 1) - tf.reduce_sum(tf.log(lengthscales))  # N,

        return kern.variance * tf.exp(-0.5 * q - tf.expand_dims(half_log_dets, 1))

# Identity mean - RBF kernel
@dispatch(mean_functions.Identity, type(None), kernels.RBF, InducingPoints, Gaussian)
def expectation(identity_mean, none, rbf_kern, feat, p):
    return tf.matrix_transpose(expectation(rbf_kern, feat, identity_mean, None, p))

# RBF kernel - Identity mean
@dispatch(kernels.RBF, InducingPoints, mean_functions.Identity, type(None), Gaussian)
def expectation(rbf_kern, feat, identity_mean, none, p):
    """
    It computes the expectation:
    <x K_{x, Z}>_p_{x},
    :return: NxMxQ
    """
    with params_as_tensors_for(feat), \
         params_as_tensors_for(rbf_kern), \
         params_as_tensors_for(identity_mean), \
         params_as_tensors_for(p):

        Xmu = p.mu
        Xcov = p.cov

        msg_input_shape = "Currently cannot handle slicing in exKxz."
        assert_input_shape = tf.assert_equal(tf.shape(Xmu)[1], rbf_kern.input_dim, message=msg_input_shape)
        assert_cov_shape = tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[:2], name="assert_Xmu_Xcov_shape")
        with tf.control_dependencies([assert_input_shape, assert_cov_shape]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        lengthscales = rbf_kern.lengthscales if rbf_kern.ARD else tf.zeros((D,), dtype=settings.np_float) + rbf_kern.lengthscales
        scalemat = tf.expand_dims(tf.matrix_diag(lengthscales ** 2.0), 0) + Xcov  # NxDxD

        det = tf.matrix_determinant(
            tf.expand_dims(tf.eye(tf.shape(Xmu)[1], dtype=settings.np_float), 0) +
            tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xcov)  # N

        vec = tf.expand_dims(tf.transpose(feat.Z), 0) - tf.expand_dims(Xmu, 2)  # NxDxM
        smIvec = tf.matrix_solve(scalemat, vec)  # NxDxM
        q = tf.reduce_sum(smIvec * vec, [1])  # NxM

        addvec = tf.matmul(smIvec, Xcov, transpose_a=True) + tf.expand_dims(Xmu, 1)  # NxMxD

        return rbf_kern.variance * addvec * tf.reshape(det ** -0.5, (N, 1, 1)) * tf.expand_dims(tf.exp(-0.5 * q), 2)

# Linear mean - RBF kernel
@dispatch(mean_functions.Linear, type(None), kernels.RBF, InducingPoints, Gaussian)
def expectation(linear_mean, none, rbf_kern, feat, p):
    return tf.matrix_transpose(expectation(rbf_kern, feat, linear_mean, None, p))

# RBF kernel - Linear mean
@dispatch(kernels.RBF, InducingPoints, mean_functions.Linear, type(None), Gaussian)
def expectation(rbf_kern, feat, linear_mean, none, p):
    """
    It computes the expectation:
    <m(x) K_{x, Z}>_p_{x}
    m(x) = A x + b
    :return: NxMxQ
    """
    with params_as_tensors_for(linear_mean):

        D = linear_mean.A.shape[0].value
        Q = linear_mean.A.shape[1].value

        exKxz = expectation(rbf_kern, feat, mean_functions.Identity(D, Q), None, p)
        eKxz = expectation(rbf_kern, feat, None, None, p)

        eAxKxz = tf.reduce_sum(exKxz[:, :, None, :] * tf.transpose(linear_mean.A)[None, None, :, :], axis=3)
        ebKxz = eKxz[..., None] * linear_mean.b[None, None, :]
        return eAxKxz + ebKxz


# RBF - RBF
# TODO implement for two different RBF kernels
@dispatch(kernels.RBF, InducingPoints, kernels.RBF, InducingPoints, Gaussian)
def expectation(kern1, feat1, kern2, feat2, p):
    """
    Also known as Phi_2.
    :param Z: MxD
    :param Xmu: X mean (NxD)
    :param Xcov: X covariance matrices (NxDxD)
    :return: NxMxM
    """
    print("MD RBF Psi2")

    with params_as_tensors_for(kern1), \
         params_as_tensors_for(feat1), \
         params_as_tensors_for(kern2), \
         params_as_tensors_for(feat2), \
         params_as_tensors_for(p):

        # use only active dimensions
        Xcov = kern1._slice_cov(p.cov)
        Z, Xmu = kern1._slice(feat1.Z, p.mu)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = kern1.lengthscales if kern1.ARD else tf.zeros((D,), dtype=settings.tf_float) + kern1.lengthscales

        Kmms = tf.sqrt(kern1.K(Z, presliced=True)) / kern1.variance ** 0.5
        scalemat = tf.expand_dims(tf.eye(D, dtype=settings.tf_float), 0) + 2 * Xcov * tf.reshape(lengthscales ** -2.0, [1, 1, -1])  # NxDxD
        det = tf.matrix_determinant(scalemat)

        mat = Xcov + 0.5 * tf.expand_dims(tf.matrix_diag(lengthscales ** 2.0), 0)  # NxDxD
        cm = tf.cholesky(mat)  # NxDxD
        vec = 0.5 * (tf.reshape(tf.transpose(Z), [1, D, 1, M]) +
                     tf.reshape(tf.transpose(Z), [1, D, M, 1])) - tf.reshape(Xmu, [N, D, 1, 1])  # NxDxMxM
        svec = tf.reshape(vec, (N, D, M * M))
        ssmI_z = tf.matrix_triangular_solve(cm, svec)  # NxDx(M*M)
        smI_z = tf.reshape(ssmI_z, (N, D, M, M))  # NxDxMxM
        fs = tf.reduce_sum(tf.square(smI_z), [1])  # NxMxM

        return kern1.variance ** 2.0 * tf.expand_dims(Kmms, 0) * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1])
