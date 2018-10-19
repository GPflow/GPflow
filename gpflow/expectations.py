# Copyright 2018 the GPflow authors.
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
# limitations under the License.

import functools
import warnings
import itertools as it

import numpy as np
import tensorflow as tf

from . import kernels, mean_functions, settings
from .features import InducingFeature, InducingPoints
from .decors import params_as_tensors_for
from .quadrature import mvnquad
from .probability_distributions import Gaussian, DiagonalGaussian, MarkovGaussian

from .dispatch import dispatch


logger = settings.logger()


# Sections:
# - Quadrature Expectations
# - Analytic Expectations
#   - RBF Kernel
#   - Linear Kernel
#   - exKxz transpose and mean function handling
#   - Mean Functions
#   - Sum Kernel
#   - RBF-Linear Cross Kernel Expectations
#   - Product Kernel
#   - Conversion to Gaussian from Diagonal or Markov


# ========================== QUADRATURE EXPECTATIONS ==========================

def quadrature_expectation(p, obj1, obj2=None, num_gauss_hermite_points=None):
    """
    Compute the expectation <obj1(x) obj2(x)>_p(x)
    Uses Gauss-Hermite quadrature for approximate integration.

    :type p: (mu, cov) tuple or a `ProbabilityDistribution` object
    :type obj1: kernel, mean function, (kernel, features), or None
    :type obj2: kernel, mean function, (kernel, features), or None
    :param int num_gauss_hermite_points: passed to `_quadrature_expectation` to set
                                         the number of Gauss-Hermite points used
    :return: a 1-D, 2-D, or 3-D tensor containing the expectation
    """
    if isinstance(p, tuple):
        assert len(p) == 2

        if   p[1].shape.ndims == 2:
            p = DiagonalGaussian(*p)
        elif p[1].shape.ndims == 3:
            p = Gaussian(*p)
        elif p[1].shape.ndims == 4:
            p = MarkovGaussian(*p)

    if isinstance(obj1, tuple):
        obj1, feat1 = obj1
    else:
        feat1 = None

    if isinstance(obj2, tuple):
        obj2, feat2 = obj2
    else:
        feat2 = None

    return _quadrature_expectation(p, obj1, feat1, obj2, feat2, num_gauss_hermite_points)


def get_eval_func(obj, feature, slice=np.s_[...]):
    """
    Return the function of interest (kernel or mean) for the expectation
    depending on the type of :obj: and whether any features are given
    """
    if feature is not None:
        # kernel + feature combination
        if not isinstance(feature, InducingFeature) or not isinstance(obj, kernels.Kernel):
            raise TypeError("If `feature` is supplied, `obj` must be a kernel.")
        return lambda x: tf.transpose(feature.Kuf(obj, x))[slice]
    elif isinstance(obj, mean_functions.MeanFunction):
        return lambda x: obj(x)[slice]
    elif isinstance(obj, kernels.Kernel):
        return lambda x: obj.Kdiag(x)
    else:
        raise NotImplementedError()


@dispatch((Gaussian, DiagonalGaussian),
          object, (InducingFeature, type(None)),
          object, (InducingFeature, type(None)),
          (int, type(None)))
def _quadrature_expectation(p, obj1, feature1, obj2, feature2, num_gauss_hermite_points):
    """
    General handling of quadrature expectations for Gaussians and DiagonalGaussians
    Fallback method for missing analytic expectations
    """
    num_gauss_hermite_points = 100 if num_gauss_hermite_points is None else num_gauss_hermite_points

    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
    elif obj1 is None:
        raise NotImplementedError("First object cannot be None.")
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(x) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(x))

    if isinstance(p, DiagonalGaussian):
        if isinstance(obj1, kernels.Kernel) and isinstance(obj2, kernels.Kernel) \
                and obj1.on_separate_dims(obj2):  # no joint expectations required

            eKxz1 = quadrature_expectation(p, (obj1, feature1),
                                           num_gauss_hermite_points=num_gauss_hermite_points)
            eKxz2 = quadrature_expectation(p, (obj2, feature2),
                                           num_gauss_hermite_points=num_gauss_hermite_points)
            return eKxz1[:, :, None] * eKxz2[:, None, :]

        else:
            cov = tf.matrix_diag(p.cov)
    else:
        cov = p.cov
    return mvnquad(eval_func, p.mu, cov, num_gauss_hermite_points)


@dispatch(MarkovGaussian,
          object, (InducingFeature, type(None)),
          object, (InducingFeature, type(None)),
          (int, type(None)))
def _quadrature_expectation(p, obj1, feature1, obj2, feature2, num_gauss_hermite_points):
    """
    Handling of quadrature expectations for Markov Gaussians (useful for time series)
    Fallback method for missing analytic expectations wrt Markov Gaussians
    Nota Bene: obj1 is always associated with x_n, whereas obj2 always with x_{n+1}
               if one requires e.g. <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1}), compute the
               transpose and then transpose the result of the expectation
    """
    num_gauss_hermite_points = 40 if num_gauss_hermite_points is None else num_gauss_hermite_points

    if obj2 is None:
        eval_func = lambda x: get_eval_func(obj1, feature1)(x)
        mu, cov = p.mu[:-1], p.cov[0, :-1]  # cross covariances are not needed
    elif obj1 is None:
        eval_func = lambda x: get_eval_func(obj2, feature2)(x)
        mu, cov = p.mu[1:], p.cov[0, 1:]  # cross covariances are not needed
    else:
        eval_func = lambda x: (get_eval_func(obj1, feature1, np.s_[:, :, None])(tf.split(x, 2, 1)[0]) *
                               get_eval_func(obj2, feature2, np.s_[:, None, :])(tf.split(x, 2, 1)[1]))
        mu = tf.concat((p.mu[:-1, :], p.mu[1:, :]), 1)  # Nx2D
        cov_top = tf.concat((p.cov[0, :-1, :, :], p.cov[1, :-1, :, :]), 2)  # NxDx2D
        cov_bottom = tf.concat((tf.matrix_transpose(p.cov[1, :-1, :, :]), p.cov[0, 1:, :, :]), 2)
        cov = tf.concat((cov_top, cov_bottom), 1)  # Nx2Dx2D

    return mvnquad(eval_func, mu, cov, num_gauss_hermite_points)


# =========================== ANALYTIC EXPECTATIONS ===========================

def expectation(p, obj1, obj2=None, nghp=None):
    """
    Compute the expectation <obj1(x) obj2(x)>_p(x)
    Uses multiple-dispatch to select an analytical implementation,
    if one is available. If not, it falls back to quadrature.

    :type p: (mu, cov) tuple or a `ProbabilityDistribution` object
    :type obj1: kernel, mean function, (kernel, features), or None
    :type obj2: kernel, mean function, (kernel, features), or None
    :param int nghp: passed to `_quadrature_expectation` to set the number
                     of Gauss-Hermite points used: `num_gauss_hermite_points`
    :return: a 1-D, 2-D, or 3-D tensor containing the expectation

    Allowed combinations

    - Psi statistics:
        >>> eKdiag = expectation(p, kern)  (N)  # Psi0
        >>> eKxz = expectation(p, (kern, feat))  (NxM)  # Psi1
        >>> exKxz = expectation(p, identity_mean, (kern, feat))  (NxDxM)
        >>> eKzxKxz = expectation(p, (kern, feat), (kern, feat))  (NxMxM)  # Psi2

    - kernels and mean functions:
        >>> eKzxMx = expectation(p, (kern, feat), mean)  (NxMxQ)
        >>> eMxKxz = expectation(p, mean, (kern, feat))  (NxQxM)

    - only mean functions:
        >>> eMx = expectation(p, mean)  (NxQ)
        >>> eM1x_M2x = expectation(p, mean1, mean2)  (NxQ1xQ2)
        .. note:: mean(x) is 1xQ (row vector)

    - different kernels. This occurs, for instance, when we are calculating Psi2 for Sum kernels:
        >>> eK1zxK2xz = expectation(p, (kern1, feat), (kern2, feat))  (NxMxM)
    """
    if isinstance(p, tuple):
        assert len(p) == 2

        if   p[1].shape.ndims == 2:
            p = DiagonalGaussian(*p)
        elif p[1].shape.ndims == 3:
            p = Gaussian(*p)
        elif p[1].shape.ndims == 4:
            p = MarkovGaussian(*p)

    if isinstance(obj1, tuple):
        obj1, feat1 = obj1
    else:
        feat1 = None

    if isinstance(obj2, tuple):
        obj2, feat2 = obj2
    else:
        feat2 = None

    try:
        return _expectation(p, obj1, feat1, obj2, feat2, nghp=nghp)
    except NotImplementedError as e:  # pragma: no cover

        warn_msg = "Quadrature is used to calculate the expectation. " + str(e)
        logger.warn(warn_msg)

        return _quadrature_expectation(p, obj1, feat1, obj2, feat2, nghp)


# ================================ RBF Kernel =================================

@dispatch(Gaussian, kernels.RBF, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3, nghp=None):
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: RBF kernel

    :return: N
    """
    return kern.Kdiag(p.mu)


@dispatch(Gaussian, kernels.RBF, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none1, none2, nghp=None):
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: RBF kernel

    :return: NxM
    """
    with params_as_tensors_for(kern, feat):
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


@dispatch(Gaussian, mean_functions.Identity, type(None), kernels.RBF, InducingPoints)
def _expectation(p, mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_n K_{x_n, Z}>_p(x_n)
        - K_{.,.} :: RBF kernel

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    with tf.control_dependencies([tf.assert_equal(
            tf.shape(Xmu)[1], tf.constant(kern.input_dim, settings.tf_int),
            message="Currently cannot handle slicing in exKxz.")]):
        Xmu = tf.identity(Xmu)

    with params_as_tensors_for(kern, feat):
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
        non_exponent_term = tf.expand_dims(Xmu, 2) + non_exponent_term  # NxDxM

        exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        return kern.variance * (determinants[:, None] * exponent_mahalanobis)[:, None, :] * non_exponent_term


@dispatch(MarkovGaussian, mean_functions.Identity, type(None), kernels.RBF, InducingPoints)
def _expectation(p, mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_{n+1} K_{x_n, Z}>_p(x_{n:n+1})
        - K_{.,.} :: RBF kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxDxM
    """
    Xmu, Xcov = p.mu, p.cov

    with tf.control_dependencies([tf.assert_equal(
            tf.shape(Xmu)[1], tf.constant(kern.input_dim, settings.tf_int),
            message="Currently cannot handle slicing in exKxz.")]):
        Xmu = tf.identity(Xmu)

    with params_as_tensors_for(kern, feat):
        D = tf.shape(Xmu)[1]
        lengthscales = kern.lengthscales if kern.ARD \
            else tf.zeros((D,), dtype=settings.float_type) + kern.lengthscales

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(lengthscales ** 2) + Xcov[0, :-1])  # NxDxD
        all_diffs = tf.transpose(feat.Z) - tf.expand_dims(Xmu[:-1], 2)  # NxDxM

        sqrt_det_L = tf.reduce_prod(lengthscales)
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N

        exponent_mahalanobis = tf.cholesky_solve(chol_L_plus_Xcov, all_diffs)  # NxDxM
        non_exponent_term = tf.matmul(Xcov[1, :-1], exponent_mahalanobis, transpose_a=True)
        non_exponent_term = tf.expand_dims(Xmu[1:], 2) + non_exponent_term  # NxDxM

        exponent_mahalanobis = tf.reduce_sum(all_diffs * exponent_mahalanobis, 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        return kern.variance * (determinants[:, None] * exponent_mahalanobis)[:, None, :] * non_exponent_term


@dispatch((Gaussian, DiagonalGaussian), kernels.RBF, InducingPoints, kernels.RBF, InducingPoints)
def _expectation(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: RBF kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other.

    :return: N x dim(Z1) x dim(Z2)
    """

    if kern1.on_separate_dims(kern2) and isinstance(p, DiagonalGaussian):
        # no joint expectations required
        eKxz1 = expectation(p, (kern1, feat1))
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    Ka, Kb = kern1, kern2

    with params_as_tensors_for(Ka, feat1, Kb, feat2):

        # use only active dimensions
        Xcov = Ka._slice_cov(tf.matrix_diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)

        Z1, Xmu = Ka._slice(feat1.Z, p.mu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        def get_squared_length_scales(kern):
            squared_lengthscales = kern.lengthscales ** 2. if kern.ARD \
                        else tf.zeros((D,), dtype=settings.tf_float) + kern.lengthscales ** 2.
            return squared_lengthscales

        if Ka == Kb:
            La = get_squared_length_scales(Ka)
            Lb = La
            half_mean_L = La * 0.5  # average length scale
        else:
            La, Lb = map(get_squared_length_scales, (Ka, Kb))
            half_mean_L = La * Lb / (La + Lb)  # average length scale

        sqrt_det_L = tf.reduce_prod(half_mean_L) ** 0.5
        C = tf.cholesky(tf.matrix_diag(half_mean_L) + Xcov)  # [N, D, D]
        dets = sqrt_det_L / tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(C)), axis=1))  # N

        # for mahalanobis computation we need Zᵀ (CCᵀ)⁻¹ Z  as well as C⁻¹ Z
        # with Z = Z₁, Z₂  for two rbf kernels
        def get_cholesky_solve_terms(Z, C=C):
            C_inv_z = tf.matrix_triangular_solve(
                C, tf.tile(tf.expand_dims(tf.transpose(Z), 0),
                           [N, 1, 1]), lower=True)  # [N, D, M]
            z_CC_inv_z = tf.reduce_sum(tf.square(C_inv_z), 1)  # [N, M]

            return C_inv_z, z_CC_inv_z

        C_inv_mu = tf.matrix_triangular_solve(C, tf.expand_dims(Xmu, 2), lower=True)  # [N, D, 1]
        mu_CC_inv_mu = tf.expand_dims(tf.reduce_sum(tf.square(C_inv_mu), 1), 2)  # [N, 1, 1]

        C_inv_z1, z1_CC_inv_z1 = get_cholesky_solve_terms(Z1 / La * half_mean_L)
        z1_CC_inv_mu = 2 * tf.matmul(C_inv_z1, C_inv_mu, transpose_a=True)[:, :, 0]  # [N, M1]

        if feat1 == feat2 and Ka == Kb:
            # in this case Z2==Z1 so we can reuse the Z1 terms
            C_inv_z2, z2_CC_inv_z2  = C_inv_z1, z1_CC_inv_z1
            z2_CC_inv_mu = z1_CC_inv_mu  # [N, M]
            Z2 = Z1
        else:
            # compute terms related to Z2
            Z2, _ = Kb._slice(feat2.Z, p.mu)
            C_inv_z2, z2_CC_inv_z2 = get_cholesky_solve_terms(Z2 / Lb * half_mean_L)
            z2_CC_inv_mu = 2 * tf.matmul(C_inv_z2, C_inv_mu, transpose_a=True)[:, :, 0]  # [N, M2]

        z1_CC_inv_z2 = tf.matmul(C_inv_z1, C_inv_z2, transpose_a=True)  # [N, M1, M2]

        # expand dims for broadcasting
        # along M1
        z2_CC_inv_mu = tf.expand_dims(z2_CC_inv_mu, 1)  # [N, 1, M2]
        z2_CC_inv_z2 = tf.expand_dims(z2_CC_inv_z2, 1)

        # along M2
        z1_CC_inv_mu = tf.expand_dims(z1_CC_inv_mu, 2)  # [N, M1, 1]
        z1_CC_inv_z1 = tf.expand_dims(z1_CC_inv_z1, 2)

        # expanded version of ((Z1 + Z2)-mu) (CCT)-1 ((Z1 + Z2)-mu)
        mahalanobis = mu_CC_inv_mu + z2_CC_inv_z2 + \
                      z1_CC_inv_z1 + 2 * z1_CC_inv_z2 - \
                      z1_CC_inv_mu - z2_CC_inv_mu  # [N, M1, M2]

        exp_mahalanobis = tf.exp(-0.5 * mahalanobis)  # [N, M1, M2]

        if Z1 == Z2:
            # CAVEAT : Compute sqrt(self.K(Z)) explicitly
            # to prevent automatic gradient from
            # being NaN sometimes, see pull request #615
            sqrt_exp_dist = tf.exp(-0.25 * Ka.scaled_square_dist(Z1, None))
        else:
            # Compute exp( -.5 (Z-Z')^top (L_1+L_2)^{-1} (Z-Z') )
            lengthscales_rms = tf.sqrt(La + Lb)
            Z1 = Z1 / lengthscales_rms
            Z1sqr = tf.reduce_sum(tf.square(Z1), axis=1)
            Z2 = Z2 / lengthscales_rms
            Z2sqr = tf.reduce_sum(tf.square(Z2), axis=1)
            dist = -2 * tf.matmul(Z1, Z2, transpose_b=True) \
                   + tf.reshape(Z1sqr, (-1, 1)) + tf.reshape(Z2sqr, (1, -1))
            sqrt_exp_dist = tf.exp(-0.5 * dist)  # M1 x M2

        return Ka.variance * Kb.variance * sqrt_exp_dist * \
               tf.reshape(dets, [N, 1, 1]) * exp_mahalanobis


# =============================== Linear Kernel ===============================

@dispatch(Gaussian, kernels.Linear, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3, nghp=None):
    """
    Compute the expectation:
    <diag(K_{X, X})>_p(X)
        - K_{.,.} :: Linear kernel

    :return: N
    """
    with params_as_tensors_for(kern):
        # use only active dimensions
        Xmu, _ = kern._slice(p.mu, None)
        Xcov = kern._slice_cov(p.cov)

        return tf.reduce_sum(kern.variance * (tf.matrix_diag_part(Xcov) + tf.square(Xmu)), 1)


@dispatch(Gaussian, kernels.Linear, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none1, none2, nghp=None):
    """
    Compute the expectation:
    <K_{X, Z}>_p(X)
        - K_{.,.} :: Linear kernel

    :return: NxM
    """
    with params_as_tensors_for(kern, feat):
        # use only active dimensions
        Z, Xmu = kern._slice(feat.Z, p.mu)

        return tf.matmul(Xmu, Z * kern.variance, transpose_b=True)


@dispatch(Gaussian, kernels.Linear, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, mean, none, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_n^T>_p(x_n)
        - K_{.,.} :: Linear kernel

    :return: NxMxD
    """
    Xmu, Xcov = p.mu, p.cov

    with tf.control_dependencies([tf.assert_equal(
            tf.shape(Xmu)[1], tf.constant(kern.input_dim, settings.tf_int),
            message="Currently cannot handle slicing in exKxz.")]):
        Xmu = tf.identity(Xmu)

    with params_as_tensors_for(kern, feat):
        N = tf.shape(Xmu)[0]
        var_Z = kern.variance * feat.Z  # MxD
        tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
        return tf.matmul(tiled_Z, Xcov + (Xmu[..., None] * Xmu[:, None, :]))


@dispatch(MarkovGaussian, kernels.Linear, InducingPoints, mean_functions.Identity, type(None))
def _expectation(p, kern, feat, mean, none, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} x_{n+1}^T>_p(x_{n:n+1})
        - K_{.,.} :: Linear kernel
        - p       :: MarkovGaussian distribution (p.cov 2x(N+1)xDxD)

    :return: NxMxD
    """
    Xmu, Xcov = p.mu, p.cov

    with tf.control_dependencies([tf.assert_equal(
            tf.shape(Xmu)[1], tf.constant(kern.input_dim, settings.tf_int),
            message="Currently cannot handle slicing in exKxz.")]):
        Xmu = tf.identity(Xmu)

    with params_as_tensors_for(kern, feat):
        N = tf.shape(Xmu)[0] - 1
        var_Z = kern.variance * feat.Z  # MxD
        tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
        eXX = Xcov[1, :-1] + (Xmu[:-1][..., None] * Xmu[1:][:, None, :])  # NxDxD
        return tf.matmul(tiled_Z, eXX)


@dispatch((Gaussian, DiagonalGaussian), kernels.Linear, InducingPoints, kernels.Linear, InducingPoints)
def _expectation(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - Ka_{.,.}, Kb_{.,.} :: Linear kernels
    Ka and Kb as well as Z1 and Z2 can differ from each other, but this is supported
    only if the Gaussian p is Diagonal (p.cov NxD) and Ka, Kb have disjoint active_dims
    in which case the joint expectations simplify into a product of expectations

    :return: NxMxM
    """
    if kern1.on_separate_dims(kern2) and isinstance(p, DiagonalGaussian):  # no joint expectations required
        eKxz1 = expectation(p, (kern1, feat1))
        eKxz2 = expectation(p, (kern2, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if kern1 != kern2 or feat1 != feat2:
        raise NotImplementedError("The expectation over two kernels has only an "
                                  "analytical implementation if both kernels are equal.")

    kern = kern1
    feat = feat1

    with params_as_tensors_for(kern, feat):
        # use only active dimensions
        Xcov = kern._slice_cov(tf.matrix_diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
        Z, Xmu = kern._slice(feat.Z, p.mu)

        N = tf.shape(Xmu)[0]
        var_Z = kern.variance * Z
        tiled_Z = tf.tile(tf.expand_dims(var_Z, 0), (N, 1, 1))  # NxMxD
        XX = Xcov + tf.expand_dims(Xmu, 1) * tf.expand_dims(Xmu, 2)  # NxDxD
        return tf.matmul(tf.matmul(tiled_Z, XX), tiled_Z, transpose_b=True)


# ================ exKxz transpose and mean function handling =================

@dispatch((Gaussian, MarkovGaussian),
          mean_functions.Identity, type(None),
          kernels.Linear, InducingPoints)
def _expectation(p, mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_n K_{x_n, Z}>_p(x_n)
        - K_{.,} :: Linear kernel
    or the equivalent for MarkovGaussian

    :return: NxDxM
    """
    return tf.matrix_transpose(expectation(p, (kern, feat), mean))


@dispatch((Gaussian, MarkovGaussian),
          kernels.Kernel, InducingFeature,
          mean_functions.MeanFunction, type(None))
def _expectation(p, kern, feat, mean, none, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <K_{Z, x_n} m(x_n)>_p(x_n)
    or the equivalent for MarkovGaussian

    :return: NxMxQ
    """
    return tf.matrix_transpose(expectation(p, mean, (kern, feat), nghp=nghp))


@dispatch(Gaussian, mean_functions.Constant, type(None), kernels.Kernel, InducingPoints)
def _expectation(p, constant_mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T K_{x_n, Z}>_p(x_n)
        - m(x_i) = c :: Constant function
        - K_{.,.}    :: Kernel function

    :return: NxQxM
    """
    with params_as_tensors_for(constant_mean):
        c = constant_mean(p.mu)  # NxQ
        eKxz = expectation(p, (kern, feat), nghp=nghp)  # NxM

        return c[..., None] * eKxz[:, None, :]


@dispatch(Gaussian, mean_functions.Linear, type(None), kernels.Kernel, InducingPoints)
def _expectation(p, linear_mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T K_{x_n, Z}>_p(x_n)
        - m(x_i) = A x_i + b :: Linear mean function
        - K_{.,.}            :: Kernel function

    :return: NxQxM
    """
    with params_as_tensors_for(linear_mean):
        N = p.mu.shape[0].value
        D = p.mu.shape[1].value
        exKxz = expectation(p, mean_functions.Identity(D), (kern, feat), nghp=nghp)
        eKxz = expectation(p, (kern, feat), nghp=nghp)
        eAxKxz = tf.matmul(tf.tile(linear_mean.A[None, :, :], (N, 1, 1)),
                           exKxz, transpose_a=True)
        ebKxz = linear_mean.b[None, :, None] * eKxz[:, None, :]
        return eAxKxz + ebKxz


@dispatch(Gaussian, mean_functions.Identity, type(None), kernels.Kernel, InducingPoints)
def _expectation(p, identity_mean, none, kern, feat, nghp=None):
    """
    This prevents infinite recursion for kernels that don't have specific
    implementations of _expectation(p, identity_mean, None, kern, feat).
    Recursion can arise because Identity is a subclass of Linear mean function
    so _expectation(p, linear_mean, none, kern, feat) would call itself.
    More specific signatures (e.g. (p, identity_mean, None, RBF, feat)) will
    be found and used whenever available
    """
    raise NotImplementedError


# ============================== Mean functions ===============================

@dispatch(Gaussian,
          (mean_functions.Linear, mean_functions.Constant),
          type(None), type(None), type(None))
def _expectation(p, mean, none1, none2, none3, nghp=None):
    """
    Compute the expectation:
    <m(X)>_p(X)
        - m(x) :: Linear, Identity or Constant mean function

    :return: NxQ
    """
    return mean(p.mu)


@dispatch(Gaussian,
          mean_functions.Constant, type(None),
          mean_functions.Constant, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Constant mean functions

    :return: NxQ1xQ2
    """
    return mean1(p.mu)[:, :, None] * mean2(p.mu)[:, None, :]


@dispatch(Gaussian,
          mean_functions.Constant, type(None),
          mean_functions.MeanFunction, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Constant mean function
        - m2(.) :: General mean function

    :return: NxQ1xQ2
    """
    e_mean2 = expectation(p, mean2)
    return mean1(p.mu)[:, :, None] * e_mean2[:, None, :]


@dispatch(Gaussian,
          mean_functions.MeanFunction, type(None),
          mean_functions.Constant, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: General mean function
        - m2(.) :: Constant mean function

    :return: NxQ1xQ2
    """
    e_mean1 = expectation(p, mean1)
    return e_mean1[:, :, None] * mean2(p.mu)[:, None, :]


@dispatch(Gaussian, mean_functions.Identity, type(None), mean_functions.Identity, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Identity mean functions

    :return: NxDxD
    """
    return p.cov + (p.mu[:, :, None] * p.mu[:, None, :])


@dispatch(Gaussian, mean_functions.Identity, type(None), mean_functions.Linear, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Identity mean function
        - m2(.) :: Linear mean function

    :return: NxDxQ
    """
    with params_as_tensors_for(mean2):
        N = tf.shape(p.mu)[0]
        e_xxt = p.cov + (p.mu[:, :, None] * p.mu[:, None, :])  # NxDxD
        e_xxt_A = tf.matmul(e_xxt, tf.tile(mean2.A[None, ...], (N, 1, 1)))  # NxDxQ
        e_x_bt = p.mu[:, :, None] * mean2.b[None, None, :]  # NxDxQ

        return e_xxt_A + e_x_bt


@dispatch(Gaussian, mean_functions.Linear, type(None), mean_functions.Identity, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.) :: Linear mean function
        - m2(.) :: Identity mean function

    :return: NxQxD
    """
    with params_as_tensors_for(mean1):
        N = tf.shape(p.mu)[0]
        e_xxt = p.cov + (p.mu[:, :, None] * p.mu[:, None, :])  # NxDxD
        e_A_xxt = tf.matmul(tf.tile(mean1.A[None, ...], (N, 1, 1)), e_xxt, transpose_a=True)  # NxQxD
        e_b_xt = mean1.b[None, :, None] * p.mu[:, None, :]  # NxQxD

        return e_A_xxt + e_b_xt


@dispatch(Gaussian, mean_functions.Linear, type(None), mean_functions.Linear, type(None))
def _expectation(p, mean1, none1, mean2, none2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m1(x_n)^T m2(x_n)>_p(x_n)
        - m1(.), m2(.) :: Linear mean functions

    :return: NxQ1xQ2
    """
    with params_as_tensors_for(mean1, mean2):
        e_xxt = p.cov + (p.mu[:, :, None] * p.mu[:, None, :])  # NxDxD
        e_A1t_xxt_A2 = tf.einsum("iq,nij,jz->nqz", mean1.A, e_xxt, mean2.A)  # NxQ1xQ2
        e_A1t_x_b2t = tf.einsum("iq,ni,z->nqz", mean1.A, p.mu, mean2.b)  # NxQ1xQ2
        e_b1_xt_A2 = tf.einsum("q,ni,iz->nqz", mean1.b, p.mu, mean2.A)  # NxQ1xQ2
        e_b1_b2t = mean1.b[:, None] * mean2.b[None, :]  # Q1xQ2

        return e_A1t_xxt_A2 + e_A1t_x_b2t + e_b1_xt_A2 + e_b1_b2t


# ================================ Sum kernels ================================

@dispatch(Gaussian, kernels.Sum, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3, nghp=None):
    """
    Compute the expectation:
    <\Sum_i diag(Ki_{X, X})>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: N
    """
    return functools.reduce(tf.add, [
        expectation(p, k, nghp=nghp) for k in kern.kernels])


@dispatch(Gaussian, kernels.Sum, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none2, none3, nghp=None):
    """
    Compute the expectation:
    <\Sum_i Ki_{X, Z}>_p(X)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxM
    """
    return functools.reduce(tf.add, [
        expectation(p, (k, feat), nghp=nghp) for k in kern.kernels])


@dispatch(Gaussian,
          (mean_functions.Linear, mean_functions.Identity, mean_functions.Constant),
          type(None), kernels.Sum, InducingPoints)
def _expectation(p, mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <m(x_n)^T (\Sum_i Ki_{x_n, Z})>_p(x_n)
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxQxM
    """
    return functools.reduce(tf.add, [
        expectation(p, mean, (k, feat), nghp=nghp) for k in kern.kernels])


@dispatch(MarkovGaussian, mean_functions.Identity, type(None), kernels.Sum, InducingPoints)
def _expectation(p, mean, none, kern, feat, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <x_{n+1} (\Sum_i Ki_{x_n, Z})>_p(x_{n:n+1})
        - \Sum_i Ki_{.,.} :: Sum kernel

    :return: NxDxM
    """
    return functools.reduce(tf.add, [
        expectation(p, mean, (k, feat), nghp=nghp) for k in kern.kernels])


@dispatch((Gaussian, DiagonalGaussian), kernels.Sum, InducingPoints, kernels.Sum, InducingPoints)
def _expectation(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <(\Sum_i K1_i_{Z1, x_n}) (\Sum_j K2_j_{x_n, Z2})>_p(x_n)
        - \Sum_i K1_i_{.,.}, \Sum_j K2_j_{.,.} :: Sum kernels

    :return: NxM1xM2
    """
    crossexps = []

    if kern1 == kern2 and feat1 == feat2:  # avoid duplicate computation by using transposes
        for i, k1 in enumerate(kern1.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k1, feat1), nghp=nghp))

            for k2 in kern1.kernels[:i]:
                eKK = expectation(p, (k1, feat1), (k2, feat2), nghp=nghp)
                eKK += tf.matrix_transpose(eKK)
                crossexps.append(eKK)
    else:
        for k1, k2 in it.product(kern1.kernels, kern2.kernels):
            crossexps.append(expectation(p, (k1, feat1), (k2, feat2), nghp=nghp))

    return functools.reduce(tf.add, crossexps)


# =================== Cross Kernel expectations (eK1zxK2xz) ===================

@dispatch((Gaussian, DiagonalGaussian), kernels.RBF, InducingPoints, kernels.Linear, InducingPoints)
def _expectation(p, rbf_kern, feat1, lin_kern, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: RBF kernel
        - K_rbf_{.,.} :: Linear kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_rbf have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    if rbf_kern.on_separate_dims(lin_kern) and isinstance(p, DiagonalGaussian):  # no joint expectations required
        eKxz1 = expectation(p, (rbf_kern, feat1))
        eKxz2 = expectation(p, (lin_kern, feat2))
        return eKxz1[:, :, None] * eKxz2[:, None, :]

    if feat1 != feat2:
        raise NotImplementedError("Features have to be the same for both kernels.")

    if rbf_kern.active_dims != lin_kern.active_dims:
        raise NotImplementedError("active_dims have to be the same for both kernels.")

    with params_as_tensors_for(rbf_kern, lin_kern, feat1, feat2):
        # use only active dimensions
        Xcov = rbf_kern._slice_cov(tf.matrix_diag(p.cov) if isinstance(p, DiagonalGaussian) else p.cov)
        Z, Xmu = rbf_kern._slice(feat1.Z, p.mu)

        N = tf.shape(Xmu)[0]
        D = tf.shape(Xmu)[1]

        lin_kern_variances = lin_kern.variance if lin_kern.ARD \
            else tf.zeros((D,), dtype=settings.tf_float) + lin_kern.variance

        rbf_kern_lengthscales = rbf_kern.lengthscales if rbf_kern.ARD \
            else tf.zeros((D,), dtype=settings.tf_float) + rbf_kern.lengthscales  ## Begin RBF eKxz code:

        chol_L_plus_Xcov = tf.cholesky(tf.matrix_diag(rbf_kern_lengthscales ** 2) + Xcov)  # NxDxD

        Z_transpose = tf.transpose(Z)
        all_diffs = Z_transpose - tf.expand_dims(Xmu, 2)  # NxDxM
        exponent_mahalanobis = tf.matrix_triangular_solve(chol_L_plus_Xcov, all_diffs, lower=True)  # NxDxM
        exponent_mahalanobis = tf.reduce_sum(tf.square(exponent_mahalanobis), 1)  # NxM
        exponent_mahalanobis = tf.exp(-0.5 * exponent_mahalanobis)  # NxM

        sqrt_det_L = tf.reduce_prod(rbf_kern_lengthscales)
        sqrt_det_L_plus_Xcov = tf.exp(tf.reduce_sum(tf.log(tf.matrix_diag_part(chol_L_plus_Xcov)), axis=1))
        determinants = sqrt_det_L / sqrt_det_L_plus_Xcov  # N
        eKxz_rbf = rbf_kern.variance * (determinants[:, None] * exponent_mahalanobis)  ## NxM <- End RBF eKxz code

        tiled_Z = tf.tile(tf.expand_dims(Z_transpose, 0), (N, 1, 1))  # NxDxM
        z_L_inv_Xcov = tf.matmul(tiled_Z, Xcov / rbf_kern_lengthscales[:, None] ** 2., transpose_a=True)  # NxMxD

        cross_eKzxKxz = tf.cholesky_solve(
            chol_L_plus_Xcov, (lin_kern_variances * rbf_kern_lengthscales ** 2.)[..., None] * tiled_Z)  # NxDxM

        cross_eKzxKxz = tf.matmul((z_L_inv_Xcov + Xmu[:, None, :]) * eKxz_rbf[..., None], cross_eKzxKxz)  # NxMxM
        return cross_eKzxKxz


@dispatch((Gaussian, DiagonalGaussian), kernels.Linear, InducingPoints, kernels.RBF, InducingPoints)
def _expectation(p, lin_kern, feat1, rbf_kern, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = <Ka_{Z1, x_n} Kb_{x_n, Z2}>_p(x_n)
        - K_lin_{.,.} :: Linear kernel
        - K_rbf_{.,.} :: RBF kernel
    Different Z1 and Z2 are handled if p is diagonal and K_lin and K_rbf have disjoint
    active_dims, in which case the joint expectations simplify into a product of expectations

    :return: NxM1xM2
    """
    return tf.matrix_transpose(expectation(p, (rbf_kern, feat2), (lin_kern, feat1)))


# ============================== Product kernels ==============================
# Note: product kernels are only supported if the kernels in kern.kernels act
#       on disjoint sets of active_dims and the Gaussian we are integrating over
#       is Diagonal

@dispatch(DiagonalGaussian, kernels.Product, type(None), type(None), type(None))
def _expectation(p, kern, none1, none2, none3, nghp=None):
    """
    Compute the expectation:
    <\HadamardProd_i diag(Ki_{X[:, active_dims_i], X[:, active_dims_i]})>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: N
    """
    if not kern.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions.")  # pragma: no cover

    return functools.reduce(tf.multiply, [
        expectation(p, k, nghp=nghp) for k in kern.kernels])


@dispatch(DiagonalGaussian, kernels.Product, InducingPoints, type(None), type(None))
def _expectation(p, kern, feat, none2, none3, nghp=None):
    """
    Compute the expectation:
    <\HadamardProd_i Ki_{X[:, active_dims_i], Z[:, active_dims_i]}>_p(X)
        - \HadamardProd_i Ki_{.,.} :: Product kernel
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxM
    """
    if not kern.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions.")  # pragma: no cover

    return functools.reduce(tf.multiply, [
        expectation(p, (k, feat), nghp=nghp) for k in kern.kernels])


@dispatch(DiagonalGaussian, kernels.Product, InducingPoints, kernels.Product, InducingPoints)
def _expectation(p, kern1, feat1, kern2, feat2, nghp=None):
    """
    Compute the expectation:
    expectation[n] = < prodK_{Z, x_n} prodK_{x_n, Z} >_p(x_n)
                   = < (\HadamardProd_i Ki_{Z[:, active_dims_i], x[n, active_dims_i]})  <-- Mx1
               1xM -->  (\HadamardProd_j Kj_{x[n, active_dims_j], Z[:, active_dims_j]}) >_p(x_n)  (MxM)

        - \HadamardProd_i Ki_{.,.}, \HadamardProd_j Kj_{.,.} :: Product kernels
        - p                        :: DiagonalGaussian distribution (p.cov NxD)

    :return: NxMxM
    """
    if feat1 != feat2:
        raise NotImplementedError("Different features are not supported.")
    if kern1 != kern2:
        raise NotImplementedError("Calculating the expectation over two "
                                  "different Product kernels is not supported.")

    kern = kern1
    feat = feat1

    if not kern.on_separate_dimensions:
        raise NotImplementedError(
            "Product currently needs to be defined on separate dimensions.")  # pragma: no cover

    return functools.reduce(tf.multiply, [
        expectation(p, (k, feat), (k, feat), nghp=nghp) for k in kern.kernels])


# ============== Conversion to Gaussian from Diagonal or Markov ===============
# Catching missing DiagonalGaussian implementations by converting to full Gaussian:

@dispatch(DiagonalGaussian,
          object, (InducingFeature, type(None)),
          object, (InducingFeature, type(None)))
def _expectation(p, obj1, feat1, obj2, feat2, nghp=None):
    gaussian = Gaussian(p.mu, tf.matrix_diag(p.cov))
    return expectation(gaussian, (obj1, feat1), (obj2, feat2), nghp=nghp)


# Catching missing MarkovGaussian implementations by converting to Gaussian (when indifferent):

@dispatch(MarkovGaussian,
          object, (InducingFeature, type(None)),
          object, (InducingFeature, type(None)))
def _expectation(p, obj1, feat1, obj2, feat2, nghp=None):
    """
    Nota Bene: if only one object is passed, obj1 is
    associated with x_n, whereas obj2 with x_{n+1}

    """
    if obj2 is None:
        gaussian = Gaussian(p.mu[:-1], p.cov[0, :-1])
        return expectation(gaussian, (obj1, feat1), nghp=nghp)
    elif obj1 is None:
        gaussian = Gaussian(p.mu[1:], p.cov[0, 1:])
        return expectation(gaussian, (obj2, feat2), nghp=nghp)
    else:
        return expectation(p, (obj1, feat1), (obj2, feat2), nghp=nghp)
