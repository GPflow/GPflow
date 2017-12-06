# Copyright 2017 the GPflow authors.
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
# limitations under the License.from __future__ import print_function

import functools
import itertools as it
import tensorflow as tf
from multipledispatch import dispatch

from . import kernels, mean_functions, settings
from .probability_distributions import Gaussian
from .expectations_quadrature import quadrature_fallback
from .features import InducingFeature, InducingPoints
from .decors import params_as_tensors_for


LINEAR_MEAN_FUNCTIONS = (mean_functions.Linear,
                         mean_functions.Constant)


# def expectation1(p, obj): pass
# def expectation2(p, obj1, obj2): pass

# def expectation(p, obj1, obj2=None):
#     if isinstance(obj1, tuple):
#         feat1, obj1 = obj1
#     if isinstance(obj2, tuple):
#         feat2, obj2 = obj2
#     if obj2 is None:
#         return expectation1(p, obj1, feat1)
#     else:
#         return expectation2(p, obj1, feat1, obj2, feat2)

# expectation(p, (feat, kern), mean_function)
# expectation(p, kern)
# expectation(p, kern, mean_function)

@dispatch(Gaussian, mean_functions.MeanFunction, type(None), kernels.Kernel, InducingFeature)
def expectation(p, mean, none, kern, feat):
    """
    It computes the expectation:
    <m(x) K_{x, Z}>_p(x), where

    :return: NxQxM
    """
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


@dispatch(Gaussian, (kernels.RBF, kernels.Linear), InducingPoints, mean_functions.Linear, type(None))
def expectation(p, rbf_kern, feat, linear_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x_i) = A x_i + b :: Linear mean function
        - K(.,.)             :: RBF or Linear kernel

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


@dispatch(Gaussian, (kernels.RBF, kernels.Linear), InducingPoints, mean_functions.Constant, type(None))
def expectation(p, rbf_kern, feat, constant_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x_i) = c :: Constant or Zero function
        - K(.,.)     :: RBF or Linear kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(constant_mean):
        c = constant_mean(p.mu) # N x Q
        eKxz = expectation(p, rbf_kern, feat, None, None) # N x M

        return eKxz[:, :, None] * c[:, None, :]


# RBF - Linear kernel
@dispatch(Gaussian, kernels.Linear, InducingPoints, kernels.RBF, InducingPoints)
def expectation(p, lin_kern, feat1, rbf_kern, feat2):
    return tf.matrix_transpose(expectation(p, rbf_kern, feat2, lin_kern, feat1))


# RBF - Linear kernel
@dispatch(Gaussian, kernels.RBF, InducingPoints, kernels.Linear, InducingPoints)
@quadrature_fallback
def expectation(p, rbf_kern, feat1, lin_kern, feat2):
    """
    It computes the expectation:
    <Ka_{x, Z1} Kb_{x, Z1}>_p(x), where
        - Ka_{.,.} :: RBF kernel
        - Ka_{.,.} :: Linear kernel

    Note that the Linear kernel and Linear mean function are different.

    :return: NxM1xM1
    """
    if feat1 != feat2 or lin_kern.ARD or \
        type(lin_kern.active_dims) is not slice or \
        type(rbf_kern.active_dims) is not slice:

            raise NotImplementedError("Active dims and/or Linear ARD not implemented. ")

    with params_as_tensors_for(feat1), \
         params_as_tensors_for(feat2), \
         params_as_tensors_for(rbf_kern), \
         params_as_tensors_for(lin_kern):

        Xmu = p.mu
        Xcov = p.cov

        Xcov = rbf_kern._slice_cov(Xcov)
        Z, Xmu = rbf_kern._slice(feat1.Z, Xmu)
        lin, rbf = lin_kern, rbf_kern

        D = tf.shape(Xmu)[1]
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]

        if rbf.ARD:
            lengthscales = rbf.lengthscales
        else:
            lengthscales = tf.zeros((D, ), dtype=settings.tf_float) + rbf.lengthscales

        lengthscales2 = lengthscales ** 2.0
        const = rbf.variance * lin.variance * tf.reduce_prod(lengthscales)
        gaussmat = Xcov + tf.matrix_diag(lengthscales2)[None, :, :]  # NxDxD
        det = tf.matrix_determinant(gaussmat) ** -0.5  # N

        cgm = tf.cholesky(gaussmat)  # NxDxD
        tcgm = tf.tile(cgm[:, None, :, :], [1, M, 1, 1])
        vecmin = Z[None, :, :] - Xmu[:, None, :]  # NxMxD
        d = tf.matrix_triangular_solve(tcgm, vecmin[:, :, :, None])  # NxMxDx1
        exp = tf.exp(-0.5 * tf.reduce_sum(d ** 2.0, [2, 3]))  # NxM

        vecplus = (Z[None, :, :, None] / lengthscales2[None, None, :, None] +
                   tf.matrix_solve(Xcov, Xmu[:, :, None])[:, None, :, :])  # NxMxDx1
        mean = tf.cholesky_solve(
            tcgm, tf.matmul(tf.tile(Xcov[:, None, :, :], [1, M, 1, 1]), vecplus))
        mean = mean[:, :, :, 0] * lengthscales2[None, None, :]  # NxMxD
        a = tf.matmul(tf.tile(Z[None, :, :], [N, 1, 1]),
                      mean * exp[:, :, None] * det[:, None, None] * const, transpose_b=True)
        return a


# RBF - RBF
@dispatch(Gaussian, kernels.RBF, InducingPoints, kernels.RBF, InducingPoints)
@quadrature_fallback
def expectation(p, kern1, feat1, kern2, feat2):
    """
    It computes the expectation:
    <Ka_{Z, x} Kb_{x, Z}>_p(x), where
        - Ka(.,.)  :: RBF kernel
        - Kb(.,.)  :: RBF kernel
    Ka and Kb can have different hyperparameters (Not implemented).
    If Ka equals Kb this expression is also known as Psi2.

    :return: N x Ma x Mb
    """
    if feat1 != feat2 or kern1 != kern2:
        raise NotImplementedError

    kern = kern1
    feat = feat1

    with params_as_tensors_for(kern), \
         params_as_tensors_for(feat):

        # use only active dimensions
        Xcov = kern._slice_cov(p.cov)
        Z, Xmu = kern._slice(feat.Z, p.mu)
        M = tf.shape(Z)[0]
        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]
        lengthscales = kern.lengthscales if kern.ARD \
                        else tf.zeros((D,), dtype=settings.tf_float) + kern.lengthscales

        Kmms = tf.sqrt(kern.K(Z, presliced=True)) / kern.variance ** 0.5
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

        return (kern.variance**2 * tf.expand_dims(Kmms, 0)
                * tf.exp(-0.5 * fs) * tf.reshape(det ** -0.5, [N, 1, 1]))


@dispatch(Gaussian, kernels.Linear, type(None), type(None), type(None))
@quadrature_fallback
def expectation(p, kern, none1, none2, none3):
        if kern.ARD:
            raise NotImplementedError
        # use only active dimensions
        X, _ = kern._slice(p.mu, None)
        Xcov = kern._slice_cov(p.cov)
        with params_as_tensors_for(kern):
            return kern.variance * \
                    (tf.reduce_sum(tf.square(X), 1) + tf.reduce_sum(tf.matrix_diag_part(Xcov), 1))


@dispatch(Gaussian, kernels.Linear, InducingPoints, type(None), type(None))
@quadrature_fallback
def expectation(p, kern, feat, none1, none2):
    if kern.ARD:
        raise NotImplementedError

    with params_as_tensors_for(kern), \
            params_as_tensors_for(feat):

        # use only active dimensions
        Z, Xmu = kern._slice(feat.Z, p.mu)

        return kern.variance * tf.matmul(Xmu, Z, transpose_b=True)


@dispatch(Gaussian, kernels.Linear, InducingPoints, kernels.Linear, InducingPoints)
@quadrature_fallback
def expectation(p, kern1, feat1, kern2, feat2):
    if kern1 != kern2 or feat1 != feat2 or kern1.ARD or kern2.ARD:
        raise NotImplementedError

    kern = kern1
    feat = feat1

    with params_as_tensors_for(kern), \
         params_as_tensors_for(feat):

        # use only active dimensions
        Xcov = kern._slice_cov(p.cov)
        Z, Xmu = kern._slice(feat.Z, p.mu)
        N = tf.shape(Xmu)[0]
        mom2 = tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2) + Xcov  # NxDxD
        eZ = tf.tile(tf.expand_dims(Z, 0), (N, 1, 1))  # NxMxD
        return kern.variance ** 2.0 * tf.matmul(tf.matmul(eZ, mom2), eZ, transpose_b=True)


# TODO: needs conversion to new expectations framework
# @params_as_tensors
# def exKxz_pairwise(self, Z, Xmu, Xcov):
#     with tf.control_dependencies([
#         tf.assert_equal(tf.shape(Xmu)[1], tf.constant(self.input_dim, settings.tf_int),
#                         message="Currently cannot handle slicing in exKxz."),
#         tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
#     ]):
#         Xmu = tf.identity(Xmu)
#
#     N = tf.shape(Xmu)[0] - 1
#     Xmum = Xmu[:-1, :]
#     Xmup = Xmu[1:, :]
#     op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
#     return self.variance * tf.matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)
#

# Linear kernel - Identity mean
@dispatch(Gaussian, kernels.Linear, InducingPoints, mean_functions.Identity, type(None))
def expectation(p, lin_kern, feat, identity_mean, none):
    """
    It computes the expectation:
    <K_{x, Z} m(x)>_p(x), where
        - m(x) = x :: identity mean function
        - K(.,.)   :: Linear kernel

    :return: NxMxQ
    """
    Xmu, Xcov = p.mu, p.cov

    with tf.control_dependencies([
        tf.assert_equal(tf.shape(Xmu)[1], tf.constant(lin_kern.input_dim, settings.np_int),
                        message="Currently cannot handle slicing in exKxz."),
        tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[:2], name="assert_Xmu_Xcov_shape")
    ]):
        Xmu = tf.identity(Xmu)

    print("Linear kernel - Identity mean")

    with params_as_tensors_for(lin_kern), \
         params_as_tensors_for(identity_mean), \
         params_as_tensors_for(feat):

        N = tf.shape(Xmu)[0]
        op = tf.expand_dims(Xmu, 2) * tf.expand_dims(Xmu, 1) + Xcov  # NxDxD
        return lin_kern.variance * tf.matmul(tf.tile(tf.expand_dims(feat.Z, 0), (N, 1, 1)), op)


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


# Sum
@dispatch(Gaussian, kernels.Sum, type(None), type(None), type(None))
def expectation(p, kern, none1, none2, none3):
    print("Sum eKdiag")
    expectation_fn = lambda k: expectation(p, k, None, None, None)
    return functools.reduce(tf.add, [expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Sum, InducingPoints, type(None), type(None))
def expectation(p, kern, feat, none2, none3):
    expectation_fn = lambda k: expectation(p, k, feat, None, None)
    return functools.reduce(tf.add, [expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Sum, InducingPoints, LINEAR_MEAN_FUNCTIONS, type(None))
def expectation(p, kern, feat, mean, none3):
    expectation_fn = lambda k: expectation(p, k, feat, mean, None)
    return functools.reduce(tf.add, [expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Sum, InducingPoints, kernels.Sum, InducingPoints)
@quadrature_fallback
def expectation(p, kern1, feat1, kern2, feat2):
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported")

    feat = feat1
    crossexps = []

    for k1, k2 in it.product(kern1.kern_list, kern2.kern_list):
        if k1.on_seperate_dims(k2):
            eKxz1 = expectation(p, k1, feat, None, None)
            eKxz2 = expectation(p, k2, feat, None, None)
            result = eKxz1[:, :, None] * eKxz2[:, None, :]
        else:
            result = expectation(p, k1, feat, k2, feat)

        crossexps.append(result)

    return functools.reduce(tf.add, crossexps)


# Product
@dispatch(Gaussian, kernels.Product, type(None), type(None), type(None))
def expectation(p, kern, none1, none2, none3):
    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.cov), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        expectation_fn = lambda k: expectation(p, k, None, None, None)
        return functools.reduce(tf.multiply, [expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Product, InducingPoints, type(None), type(None))
def expectation(p, kern, feat, none2, none3):
    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.cov), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        expectation_fn = lambda k: expectation(p, k, feat, None, None)
        return functools.reduce(tf.multiply, [expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Product, InducingPoints, kernels.Product, InducingPoints)
@quadrature_fallback
def expectation(p, kern1, feat1, kern2, feat2):
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported")

    if kern1 != kern2:
        raise NotImplementedError("Calculating the expectation over two different Product kernels is not supported")

    kern = kern1
    feat = feat1

    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.cov), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        expectation_fn = lambda k: expectation(p, k, feat, k, feat)
        return functools.reduce(tf.multiply, [expectation_fn(k) for k in kern.kern_list])
