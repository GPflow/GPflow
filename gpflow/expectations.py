import types

import numpy as np
import tensorflow as tf
from multipledispatch import dispatch

from . import kernels, mean_functions, settings
from .features import InducingFeature, InducingPoints
from .quadrature import mvnquad
from .decors import params_as_tensors_for
from .params import Parameterized, Parameter


LINEAR_MEAN_FUNCTIONS = (mean_functions.Linear,
                         mean_functions.Constant)


class ProbabilityDistribution():
    pass


class Gaussian(ProbabilityDistribution):
    def __init__(self, mu, cov):
        super(Gaussian, self).__init__()
        self.mu = mu  # N x D
        self.cov = cov  # N x D x D


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


@dispatch(Gaussian, object, (InducingFeature, type(None)), object, (InducingFeature, type(None)))
def expectation(p, obj1, feature1, obj2, feature2, H=20):
    print("Quad")
    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature1)(x)
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    return mvnquad(eval_func, p.mu, p.cov, H)


@dispatch(Gaussian, mean_functions.MeanFunction, type(None), kernels.Kern, InducingFeature)
def expectation(p, mean, none, kern, feat):
    """
    It computes the expectation:
    <m(x) K_{x, Z}>_p(x), where

    :return: NxQxM
    """
    print("Expectation: Mean - Kern")
    return tf.matrix_transpose(expectation(p, kern, feat, mean, None))


@dispatch(Gaussian, kernels.RBF, type(None), type(None), type(None))
def expectation(p, kern, none1, none2, none3):
    """
    It computes the expectation:
    <K_{x, x}>_p(x), where
        - K(.,.)   :: RBF kernel
    This is expression is also known is Psi0

    :return: N
    """
    return kern.Kdiag(p.mu)


@dispatch(Gaussian, kernels.RBF, InducingPoints, type(None), type(None))
def expectation(p, kern, feat, none1, none2):
    """
    It computes the expectation:
    <K_{x, Z}>_p(x), where
        - K(.,.)   :: RBF kernel
    This is expression is also known is Psi1

    :return: NxM
    """
    with params_as_tensors_for(feat), \
         params_as_tensors_for(kern):

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
        half_log_dets = (tf.reduce_sum(tf.log(chol_diags), 1)
                         - tf.reduce_sum(tf.log(lengthscales)))  # N,

        return kern.variance * tf.exp(-0.5 * q - tf.expand_dims(half_log_dets, 1))


# RBF kernel - Identity mean
@dispatch(Gaussian, kernels.RBF, InducingPoints, mean_functions.Identity, type(None))
def expectation(p, rbf_kern, feat, identity_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x) = x :: identity mean function
        - K(.,.)   :: RBF kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(feat), \
         params_as_tensors_for(rbf_kern), \
         params_as_tensors_for(identity_mean):

        Xmu = p.mu
        Xcov = p.cov

        msg_input_shape = "Currently cannot handle slicing in exKxz."
        assert_input_shape = tf.assert_equal(tf.shape(Xmu)[1],
                                             rbf_kern.input_dim, message=msg_input_shape)
        assert_cov_shape = tf.assert_equal(tf.shape(Xmu),
                                           tf.shape(Xcov)[:2], name="assert_Xmu_Xcov_shape")
        with tf.control_dependencies([assert_input_shape, assert_cov_shape]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        lengthscales = rbf_kern.lengthscales if rbf_kern.ARD \
                        else tf.zeros((D,), dtype=settings.np_float) + rbf_kern.lengthscales
        scalemat = tf.expand_dims(tf.matrix_diag(lengthscales ** 2.0), 0) + Xcov  # NxDxD

        det = tf.matrix_determinant(
            tf.expand_dims(tf.eye(tf.shape(Xmu)[1], dtype=settings.np_float), 0) +
            tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xcov)  # N

        vec = tf.expand_dims(tf.transpose(feat.Z), 0) - tf.expand_dims(Xmu, 2)  # NxDxM
        smIvec = tf.matrix_solve(scalemat, vec)  # NxDxM
        q = tf.reduce_sum(smIvec * vec, [1])  # NxM

        addvec = tf.matmul(smIvec, Xcov, transpose_a=True) + tf.expand_dims(Xmu, 1)  # NxMxD

        return (rbf_kern.variance * addvec * tf.reshape(det ** -0.5, (N, 1, 1))
                * tf.expand_dims(tf.exp(-0.5 * q), 2))


# RBF kernel - Linear mean
@dispatch(Gaussian, kernels.RBF, InducingPoints, mean_functions.Linear, type(None))
def expectation(p, rbf_kern, feat, linear_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x_i) = A x_i + b :: Linear mean function
        - K(.,.)             :: RBF kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(linear_mean):
        D_in = p.mu.shape[1].value

        exKxz = expectation(p, rbf_kern, feat, mean_functions.Identity(D_in), None)
        eKxz = expectation(p, rbf_kern, feat, None, None)

        eAxKxz = tf.reduce_sum(exKxz[:, :, None, :]
                               * tf.transpose(linear_mean.A)[None, None, :, :], axis=3)
        ebKxz = eKxz[..., None] * linear_mean.b[None, None, :]
        return eAxKxz + ebKxz


# RBF kernel - Constant or Zero mean
@dispatch(Gaussian, kernels.RBF, InducingPoints,
          (mean_functions.Zero, mean_functions.Constant), type(None))
def expectation(p, rbf_kern, feat, constant_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x_i) = A x_i + b :: Constant or Zero function
        - K(.,.)             :: RBF kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(constant_mean):
        c = constant_mean(p.mu) # N x Q
        eKxz = expectation(p, rbf_kern, feat, None, None) # N x M

        return eKxz[:, :, None] * c[:, None, :]


# RBF - RBF
# TODO implement for two different RBF kernels
@dispatch(Gaussian, kernels.RBF, InducingPoints, kernels.RBF, InducingPoints)
def expectation(p, kern1, feat1, kern2, feat2):
    """
    It computes the expectation:
    <Ka_{Z, x} Kb_{x, Z}>_p(x), where
        - Ka(.,.)  :: RBF kernel
        - Kb(.,.)  :: RBF kernel
    Ka and Kb can have different hyperparameters.
    If Ka equals Kb this expression is also known as Psi2.

    :return: N x Ma x Mb
    """
    with params_as_tensors_for(kern1), \
         params_as_tensors_for(feat1), \
         params_as_tensors_for(kern2), \
         params_as_tensors_for(feat2):

        # use only active dimensions
        Xcov = kern1._slice_cov(p.cov)
        Z, Xmu = kern1._slice(feat1.Z, p.mu)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = kern1.lengthscales if kern1.ARD \
                        else tf.zeros((D,), dtype=settings.tf_float) + kern1.lengthscales

        Kmms = tf.sqrt(kern1.K(Z, presliced=True)) / kern1.variance ** 0.5
        scalemat = (tf.expand_dims(tf.eye(D, dtype=settings.tf_float), 0)
                    + 2 * Xcov * tf.reshape(lengthscales ** -2.0, [1, 1, -1]))  # NxDxD
        det = tf.matrix_determinant(scalemat)

        mat = Xcov + 0.5 * tf.expand_dims(tf.matrix_diag(lengthscales ** 2.0), 0)  # NxDxD
        cm = tf.cholesky(mat)  # NxDxD
        vec = 0.5 * (tf.reshape(tf.transpose(Z), [1, D, 1, M]) +
                     tf.reshape(tf.transpose(Z), [1, D, M, 1])) - tf.reshape(Xmu, [N, D, 1, 1])  # NxDxMxM
        svec = tf.reshape(vec, (N, D, M * M))
        ssmI_z = tf.matrix_triangular_solve(cm, svec)  # NxDx(M*M)
        smI_z = tf.reshape(ssmI_z, (N, D, M, M))  # NxDxMxM
        fs = tf.reduce_sum(tf.square(smI_z), [1])  # NxMxM

        return (kern1.variance**2 * tf.expand_dims(Kmms, 0)
                * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1]))


@dispatch(Gaussian, LINEAR_MEAN_FUNCTIONS, type(None), type(None), type(None))
def expectation(p, mean, none1, none2, none3):
    return mean(p.mu)


@dispatch(Gaussian, mean_functions.Linear, type(None), mean_functions.Linear, type(None))
def expectation(p, mean1, none1, mean2, none2):
    with params_as_tensors_for(mean1), params_as_tensors_for(mean2):
        e_xt_x = p.cov + (p.mu[:, :, None] * p.mu[:, None, :]) # N x D x D
        e_A1t_xt_x_A2 = tf.einsum("iq,nij,jz->nqz", mean1.A, e_xt_x, mean2.A) # N x Q1 x Q2
        e_A1t_xt_b2  = tf.einsum("iq,ni,z->nqz", mean1.A, p.mu, mean2.b) # N x Q1 x Q2
        e_b1t_x_A2  = tf.einsum("q,ni,iz->nqz", mean1.b, p.mu, mean2.A) # N x Q1 x Q2
        e_b1t_b2 = mean1.b[:, None] * mean2.b[None, :] # Q1 x Q2

        return e_A1t_xt_x_A2 + e_A1t_xt_b2 + e_b1t_x_A2 + e_b1t_b2


@dispatch(Gaussian,
          mean_functions.Constant, type(None),
          mean_functions.Constant, type(None))
def expectation(p, mean1, none1, mean2, none2):
    return mean1(p.mu)[:, :, None] * mean2(p.mu)[:, None, :]


@dispatch(Gaussian,
          mean_functions.MeanFunction, type(None),
          mean_functions.Constant, type(None))
def expectation(p, mean1, none1, mean2, none2):
    return tf.matrix_transpose(expectation(p, mean2, None, mean1, None))


@dispatch(Gaussian,
          mean_functions.Constant, type(None),
          mean_functions.MeanFunction, type(None))
def expectation(p, mean1, none1, mean2, none2):
    e_mean2 = expectation(p, mean2, None, None, None)
    return mean1(p.mu)[:, :, None] * e_mean2[:, None, :]


