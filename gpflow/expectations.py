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

from . import kernels, mean_functions, settings
from .probability_distributions import Gaussian, DiagonalGaussian, MarkovGaussian
from .expectations_quadrature import dispatch, quadrature_fallback
from .features import InducingFeature, InducingPoints
from .decors import params_as_tensors_for


def expectation(p, obj1, obj2=None):
    """
    Calculates the expectation <obj1(x) obj2(x)>_p(x)
    obj1 and obj2 can be kernels, mean functions or None.

    Using the multiple-dispatch paradigm the function will select an analytical implementation,
    if one is available to calculate the expectation, or fall back to a quadrature.

    The new framework makes the ``ekernels`` classes obsolete.

    A couple of examples:
        .. Psi statistics
        eKdiag = expectation(pX, kern)  # psi0
        eKxz = expectation(pX, (feat, kern))  # psi1
        eKzxKxz = expectation(pX, (feat, kern), (feat, kern))  # psi2

        .. kernels and mean functions
        eKzxMx = expectation(pX, (feat, kern), mean)
        eMxKxz = expectation(pX, mean, (feat, kern))

        .. only mean functions
        eMx = expectation(pX, mean)
        eMx_sq = expectation(pX, mean, mean)

        .. different kernels
        .. this occurs when we are calculating Psi2 for Sum kernels
        eK1zxK2xz = expectation(pX, (feat, kern1), (feat, kern2)) # different kernel
    """
    if isinstance(obj1, tuple):
        feat1, obj1 = obj1
    else:
        feat1 = None

    if isinstance(obj2, tuple):
        feat2, obj2 = obj2
    else:
        feat2 = None

    return _expectation(p, obj1, feat1, obj2, feat2)


@dispatch(Gaussian, mean_functions.MeanFunction, type(None), kernels.Kernel, InducingFeature)
def _expectation(p, mean, none, kern, feat):
    """
    It computes the expectation:
    expectation[n] = <m(x_n)^T K_{x_n, Z}>_p(x_n)

    :return: NxQxM
    """
    return tf.matrix_transpose(_expectation(p, kern, feat, mean, None))


@dispatch(Gaussian, kernels.RBF, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3):
    """
    It computes the expectation:
    <diag(K_{X, X})>_p(X), where
        - K(.,.)   :: RBF kernel
    This expression is also known as Psi0

    :return: N
    """
    return kern.Kdiag(p.mu)


@dispatch(Gaussian, kernels.RBF, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none1, none2):
    """
    It computes the expectation:
    <K_{X, Z}>_p(X), where
        - K(.,.)   :: RBF kernel
    This expression is also known as Psi1

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

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales ** 2) + Xcov)  # NxDxD

        all_diffs = tf.transpose(Z) - tf.expand_dims(Xmu, 2)  # NxDxM
        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        sqrt_det_L = tf.reduce_prod(lengthscales)
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        return kern.variance * (determinants[:, None] * exponent_mahalanobis)


@dispatch(Gaussian, kernels.RBF, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, identity_mean, none):
    """
    It computes the expectation:
    expectation[n] = <K_{Z, x_n} m(x_n)>_p(x_n), where
        - m(x) = x :: identity mean function
        - K(.,.)   :: RBF kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(feat), \
         params_as_tensors_for(kern), \
         params_as_tensors_for(identity_mean):

        Xmu = p.mu
        Xcov = p.cov

        msg_input_shape = "Currently cannot handle slicing in exKxz."
        assert_input_shape = tf.assert_equal(tf.shape(Xmu)[1],
                                             kern.input_dim, message=msg_input_shape)
        assert_cov_shape = tf.assert_equal(tf.shape(Xmu),
                                           tf.shape(Xcov)[:2], name="assert_Xmu_Xcov_shape")
        with tf.control_dependencies([assert_input_shape, assert_cov_shape]):
            Xmu = tf.identity(Xmu)

        D = tf.shape(Xmu)[1]
        lengthscales = kern.lengthscales if kern.ARD \
            else tf.zeros((D,), dtype=settings.float_type) + kern.lengthscales

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales ** 2) + Xcov)  # NxDxD
        all_diffs = tf.transpose(feat.Z) - tf.expand_dims(Xmu, 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales)
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
        non_exponent_term = tf.matmul(Xcov, exponent_mahalanobis, transpose_a=True)
        non_exponent_term = tf.transpose(tf.expand_dims(Xmu, 2) + non_exponent_term, [0, 2, 1])  # NxMxD

        exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        return kern.variance * (determinants[:, None] * exponent_mahalanobis)[:, :, None] * non_exponent_term


@dispatch(Gaussian, (kernels.RBF, kernels.Linear), InducingPoints, mean_functions.Linear, type(None))
def _expectation(p, kern, feat, linear_mean, none):
    """
    It computes the expectation:
    expectation[n] = <K_{Z, x_n} m(x_n)>_p(x_n), where
        - m(x_i) = A x_i + b :: Linear mean function
        - K(.,.)             :: RBF or Linear kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(linear_mean):
        D_in = p.mu.shape[1].value

        exKxz = _expectation(p, kern, feat, mean_functions.Identity(D_in), None)
        eKxz = _expectation(p, kern, feat, None, None)

        eAxKxz = tf.reduce_sum(exKxz[:, :, None, :]
                               * tf.transpose(linear_mean.A)[None, None, :, :], axis=3)
        ebKxz = eKxz[..., None] * linear_mean.b[None, None, :]
        return eAxKxz + ebKxz


@dispatch(Gaussian, (kernels.RBF, kernels.Linear), InducingPoints, mean_functions.Constant, type(None))
def _expectation(p, kern, feat, constant_mean, none):
    """
    It computes the expectation:
    expectation[n] = <K_{Z, x_n} m(x_n)>_p(x_n), where
        - m(x_i) = c :: Constant function
        - K(.,.)     :: RBF or Linear kernel

    :return: NxMxQ
    """
    with params_as_tensors_for(constant_mean):
        c = constant_mean(p.mu) # N x Q
        eKxz = _expectation(p, kern, feat, None, None) # N x M

        return eKxz[:, :, None] * c[:, None, :]


@dispatch(Gaussian, kernels.Linear, InducingPoints, kernels.RBF, InducingPoints)
def _expectation(p, lin_kern, feat1, rbf_kern, feat2):
    return tf.matrix_transpose(_expectation(p, rbf_kern, feat2, lin_kern, feat1))


@dispatch(Gaussian, kernels.RBF, InducingPoints, kernels.Linear, InducingPoints)
@quadrature_fallback
def _expectation(p, rbf_kern, feat1, lin_kern, feat2):
    """
    It computes the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n), where
        - Ka_{.,.} :: RBF kernel
        - Kb_{.,.} :: Linear kernel

    :return: NxMxM
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


@dispatch(Gaussian, kernels.RBF, InducingPoints, kernels.RBF, InducingPoints)
@quadrature_fallback
def _expectation(p, kern1, feat1, kern2, feat2):
    """
    It computes the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n), where
        - Ka(.,.) and Kb(.,.)  :: RBF kernels
    Ka and Kb can have different hyperparameters (Not implemented).
    If Ka and Kb are equal this expression is also known as Psi2.

    :return: N x Ma x Mb
    """
    if feat1 != feat2 or kern1 != kern2:
        raise NotImplementedError("The expectation over two kernels has only a "
                                  "analytical implementation if both kernels are equal.")

    kern = kern1
    feat = feat1

    with params_as_tensors_for(kern), \
         params_as_tensors_for(feat):

        # use only active dimensions
        Xcov = kern._slice_cov(p.cov)
        Z, Xmu = kern._slice(feat.Z, p.mu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        squared_lengthscales = kern.lengthscales ** 2. if kern.ARD \
            else tf.zeros((D,), dtype=settings.tf_float) + kern.lengthscales ** 2.

        sqrt_det_L = tf.reduce_prod(0.5 * squared_lengthscales) ** 0.5
        C = tf.cholesky(0.5 * tf.matrix_diag(squared_lengthscales) + Xcov)  # NxDxD
        dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(C)), axis=1))  # N

        C_inv_mu = tf.matrix_triangular_solve(C, tf.expand_dims(Xmu, 2), lower=True)  # NxDx1
        C_inv_z = tf.matrix_triangular_solve(C,
                                             tf.tile(tf.expand_dims(tf.transpose(Z) / 2., 0), [N, 1, 1]),
                                             lower=True)  # NxDxM
        mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), 1), 2)  # Nx1x1
        z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), 1)  # NxM
        zm_CC_inv_zn = tf.matmul(C_inv_z, C_inv_z, transpose_a=True)  # NxMxM
        two_z_CC_inv_mu = 2 * tf.matmul(C_inv_z, C_inv_mu, transpose_a=True)  # NxMx1

        exponent_mahalanobis = mu_CC_inv_mu + tf.expand_dims(z_CC_inv_z, 1) + tf.expand_dims(z_CC_inv_z, 2) + \
                               2 * zm_CC_inv_zn - two_z_CC_inv_mu - tf.transpose(two_z_CC_inv_mu, [0, 2, 1])  # NxMxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxMxM

        return kern.variance ** 1.5 * tf.sqrt(kern.K(Z, presliced=True)) * \
               tf.reshape(dets, [N, 1, 1]) * exponent_mahalanobis


@dispatch(Gaussian, kernels.Linear, type(None), type(None), type(None))
@quadrature_fallback
def _expectation(p, kern, none1, none2, none3):
    """
    It computes the expectation:
    <diag(K_{X, X})>_p(X), where
        - K(.,.)   :: Linear kernel
    This expression is also known as Psi0

    :return: N
    """
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
def _expectation(p, kern, feat, none1, none2):
    """
    It computes the expectation:
    <K_{X, Z}>_p(X), where
        - K(.,.)   :: Linear kernel
    This expression is also known as Psi1

    :return: NxM
    """
    if kern.ARD:
        raise NotImplementedError

    with params_as_tensors_for(kern), \
            params_as_tensors_for(feat):

        # use only active dimensions
        Z, Xmu = kern._slice(feat.Z, p.mu)

        return kern.variance * tf.matmul(Xmu, Z, transpose_b=True)


@dispatch(Gaussian, kernels.Linear, InducingPoints, kernels.Linear, InducingPoints)
@quadrature_fallback
def _expectation(p, kern1, feat1, kern2, feat2):
    """
    It computes the expectation:
    expectation[n] = <Ka_{Z, x_n} Kb_{x_n, Z}>_p(x_n), where
        - Ka(.,.) and Kb(.,.)   :: Linear kernels
    This expression is also known as Psi2

    :return: NxMxM
    """
    if kern1 != kern2 or feat1 != feat2 or kern1.ARD or kern2.ARD:
        raise NotImplementedError("The expectation over two kernels has only a "
                                  "analytical implementation if both kernels are equal.")

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


@dispatch(Gaussian, kernels.Linear, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, lin_kern, feat, identity_mean, none):
    """
    It computes the expectation:
    expectation[n] = <K_{Z, x_n} m(x)>_p(x_n), where
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

    with params_as_tensors_for(lin_kern), \
         params_as_tensors_for(identity_mean), \
         params_as_tensors_for(feat):

        N = tf.shape(Xmu)[0]
        op = tf.expand_dims(Xmu, 2) * tf.expand_dims(Xmu, 1) + Xcov  # NxDxD
        return lin_kern.variance * tf.matmul(tf.tile(tf.expand_dims(feat.Z, 0), (N, 1, 1)), op)


@dispatch(Gaussian, (mean_functions.Linear, mean_functions.Constant), type(None), type(None), type(None))
def _expectation(p, mean, none1, none2, none3):
    """
    It computes the expectation:
    <m(X)>_p(X), where
        - m(x) :: Linear or Constant mean function

    :return: NxQ
    """
    return mean(p.mu)


@dispatch(Gaussian, mean_functions.Linear, type(None), mean_functions.Linear, type(None))
def _expectation(p, mean1, none1, mean2, none2):
    """
    It computes the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n), where
        - m1(.), m2(.) :: Linear mean functions

    :return: NxQxQ
    """
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
def _expectation(p, mean1, none1, mean2, none2):
    return mean1(p.mu)[:, :, None] * mean2(p.mu)[:, None, :]


@dispatch(Gaussian,
          mean_functions.MeanFunction, type(None),
          mean_functions.Constant, type(None))
def _expectation(p, mean1, none1, mean2, none2):
    return tf.matrix_transpose(_expectation(p, mean2, None, mean1, None))


@dispatch(Gaussian,
          mean_functions.Constant, type(None),
          mean_functions.MeanFunction, type(None))
def _expectation(p, mean1, none1, mean2, none2):
    e_mean2 = _expectation(p, mean2, None, None, None)
    return mean1(p.mu)[:, :, None] * e_mean2[:, None, :]


# Sum
@dispatch(Gaussian, kernels.Sum, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3):
    _expectation_fn = lambda k: _expectation(p, k, None, None, None)
    return functools.reduce(tf.add, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Sum, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none2, none3):
    _expectation_fn = lambda k: _expectation(p, k, feat, None, None)
    return functools.reduce(tf.add, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian,
          kernels.Sum, InducingPoints,
          (mean_functions.Linear, mean_functions.Constant), type(None))
def _expectation(p, kern, feat, mean, none3):
    _expectation_fn = lambda k: _expectation(p, k, feat, mean, None)
    return functools.reduce(tf.add, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(MarkovGaussian, kernels.Sum, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, mean, none3):
    _expectation_fn = lambda k: _expectation(p, k, feat, mean, None)
    return functools.reduce(tf.add, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(Gaussian, kernels.Sum, InducingPoints, kernels.Sum, InducingPoints)
@quadrature_fallback
def _expectation(p, kern1, feat1, kern2, feat2):
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported")

    feat = feat1
    crossexps = []

    for k1, k2 in it.product(kern1.kern_list, kern2.kern_list):
        if k1.on_separate_dims(k2):
            eKxz1 = _expectation(p, k1, feat, None, None)
            eKxz2 = _expectation(p, k2, feat, None, None)
            result = eKxz1[:, :, None] * eKxz2[:, None, :]
        else:
            result = _expectation(p, k1, feat, k2, feat)

        crossexps.append(result)

    return functools.reduce(tf.add, crossexps)


# Product
# Note: product kernels only support Diagonal Gaussian distributions, p.cov having shape NxD
@dispatch(DiagonalGaussian, kernels.Product, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3):
    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.var), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        _expectation_fn = lambda k: _expectation(p, k, None, None, None)
        return functools.reduce(tf.multiply, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(DiagonalGaussian, kernels.Product, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none2, none3):
    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.var), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        _expectation_fn = lambda k: _expectation(p, k, feat, None, None)
        return functools.reduce(tf.multiply, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(DiagonalGaussian, kernels.Product, InducingPoints, kernels.Product, InducingPoints)
@quadrature_fallback
def _expectation(p, kern1, feat1, kern2, feat2):
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported")

    if kern1 != kern2:
        raise NotImplementedError("Calculating the expectation over two different Product kernels is not supported")

    kern = kern1
    feat = feat1

    if not kern.on_separate_dimensions:
        raise NotImplementedError("Product currently needs to be defined on separate dimensions.")  # pragma: no cover
    with tf.control_dependencies([
        tf.assert_equal(tf.rank(p.var), 2,
                        message="Product currently only supports diagonal Xcov.", name="assert_Xcov_diag"),
    ]):
        _expectation_fn = lambda k: _expectation(p, k, feat, k, feat)
        return functools.reduce(tf.multiply, [_expectation_fn(k) for k in kern.kern_list])


@dispatch(DiagonalGaussian, object, (InducingFeature, type(None)), object, (InducingFeature, type(None)))
def _expectation(p, obj1, obj2, obj3, obj4):
    gauss = Gaussian(p.mu, tf.matrix_diag(p.var))
    return _expectation(gauss, obj1, obj2, obj3, obj4)


# Time Series expectations
@dispatch(MarkovGaussian, kernels.RBF, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, mean, none):
    """
    expectation[n] = <K_{Z, x_{n-1} x_n^T}>_p(x_{n-1:n})
        - K(.,.) :: RBF kernel
        - p :: TimeSeriesGaussian distribution

    :return: NxMxQ
    """
    with params_as_tensors_for(kern), \
         params_as_tensors_for(feat):

        Xmu = p.mu
        Xcov = p.cov
        Z = feat.Z

        msg_input_shape = "Currently cannot handle slicing in exKxz_pairwise."
        assert_input_shape = tf.assert_equal(tf.shape(Xmu)[1], kern.input_dim, message=msg_input_shape)
        assert_cov_shape = tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        with tf.control_dependencies([assert_input_shape, assert_cov_shape]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        D = tf.shape(Xmu)[1]
        Xsigmb = tf.slice(Xcov, [0, 0, 0, 0], tf.stack([-1, N, -1, -1]))
        Xsigm = Xsigmb[0, :, :, :]  # NxDxD
        Xsigmc = Xsigmb[1, :, :, :]  # NxDxD
        Xmum = tf.slice(Xmu, [0, 0], tf.stack([N, -1]))
        Xmup = Xmu[1:, :]
        lengthscales = kern.lengthscales if kern.ARD else tf.zeros((D,), dtype=settings.float_type) + kern.lengthscales
        scalemat = tf.expand_dims(tf.matrix_diag(lengthscales ** 2.0), 0) + Xsigm  # NxDxD

        det = tf.matrix_determinant(
            tf.expand_dims(tf.eye(tf.shape(Xmu)[1], dtype=settings.float_type), 0) +
            tf.reshape(lengthscales ** -2.0, (1, 1, -1)) * Xsigm)  # N

        vec = tf.expand_dims(tf.transpose(Z), 0) - tf.expand_dims(Xmum, 2)  # NxDxM
        smIvec = tf.matrix_solve(scalemat, vec)  # NxDxM
        q = tf.reduce_sum(smIvec * vec, [1])  # NxM

        addvec = tf.matmul(smIvec, Xsigmc, transpose_a=True) + tf.expand_dims(Xmup, 1)  # NxMxD

        return kern.variance * addvec * tf.reshape(det ** -0.5, (N, 1, 1)) * tf.expand_dims(tf.exp(-0.5 * q), 2)


@dispatch(MarkovGaussian, kernels.Linear, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, mean, none):
    """
    expectation[n] = <K_{Z, x_{n-1}} x_n^T>_p(x_{n-1:n})
        - K(.,.) :: Linear kernel
        - p :: MarkovGaussian distribution

    :return: NxMxQ
    """
    with params_as_tensors_for(kern), \
         params_as_tensors_for(feat):

        Xmu = p.mu
        Xcov = p.cov
        Z = feat.Z

        with tf.control_dependencies([
            tf.assert_equal(tf.shape(Xmu)[1], tf.constant(kern.input_dim, settings.tf_int),
                            message="Currently cannot handle slicing in exKxz."),
            tf.assert_equal(tf.shape(Xmu), tf.shape(Xcov)[1:3], name="assert_Xmu_Xcov_shape")
        ]):
            Xmu = tf.identity(Xmu)

        N = tf.shape(Xmu)[0] - 1
        Xmum = Xmu[:-1, :]
        Xmup = Xmu[1:, :]
        op = tf.expand_dims(Xmum, 2) * tf.expand_dims(Xmup, 1) + Xcov[1, :-1, :, :]  # NxDxD
        return kern.variance * tf.matmul(tf.tile(tf.expand_dims(Z, 0), (N, 1, 1)), op)
