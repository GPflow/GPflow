# Copyright 2016-2020 The GPflow Contributors. All Rights Reserved.
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

from typing import Tuple

import numpy as np
import tensorflow as tf

from . import covariances, kernels
from .base import Module, Parameter
from .conditionals.util import expand_independent_outputs, mix_latent_gp
from .config import default_jitter, default_float
from .models.model import MeanAndVariance


class DiagNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [M, L]


class MvnNormal(Module):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [L, M, M], lower-triangular


class Posterior(Module):
    def __init__(self, kernel, iv, q_mu, q_sqrt, whiten=True, mean_function=None):
        self.iv = iv
        self.kernel = kernel
        self.mean_function = mean_function
        self.whiten = whiten
        if len(q_sqrt.shape) == 2:  # q_diag
            self.q_dist = DiagNormal(q_mu, q_sqrt)
        else:
            self.q_dist = MvnNormal(q_mu, q_sqrt)

        self.update_cache()  # populates or updates self.alpha and self.Qinv

    def _precompute(self):
        Kuu = covariances.Kuu(self.iv, self.kernel, jitter=default_jitter())  # [(R), M, M]
        q_mu = self.q_dist.q_mu

        if Kuu.shape.ndims == 4:
            ML = tf.cast(tf.sqrt(tf.cast(tf.reduce_prod(tf.shape(Kuu)), default_float())), tf.int32)
            Kuu = tf.reshape(Kuu, [ML, ML])
        if Kuu.shape.ndims == 3:
            q_mu = tf.linalg.adjoint(self.q_dist.q_mu)[..., None]  # [..., R, M, 1]
        L = tf.linalg.cholesky(Kuu)

        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = tf.linalg.cholesky_solve(L, q_mu)
        else:
            # alpha = L⁻T q_mu
            alpha = tf.linalg.triangular_solve(L, q_mu, adjoint=True)
        # predictive mean = Kfu alpha
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtT
        if not self.whiten:
            # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
            #      = Kuu⁻¹ - L⁻T L⁻¹ S L⁻T L⁻¹
            #      = L⁻T (I - L⁻¹ S L⁻T) L⁻¹
            #      = L⁻T B L⁻¹
            if isinstance(self.q_dist, DiagNormal):
                q_sqrt = tf.linalg.diag(tf.linalg.adjoint(self.q_dist.q_sqrt))
            else:
                q_sqrt = self.q_dist.q_sqrt
            Linv_qsqrt = tf.linalg.triangular_solve(L, q_sqrt)
            Linv_cov_u_LinvT = tf.matmul(Linv_qsqrt, Linv_qsqrt, transpose_b=True)
        else:
            if isinstance(self.q_dist, DiagNormal):
                Linv_cov_u_LinvT = tf.linalg.diag(tf.linalg.adjoint(self.q_dist.q_sqrt ** 2))
            else:
                q_sqrt = self.q_dist.q_sqrt
                Linv_cov_u_LinvT = tf.matmul(q_sqrt, q_sqrt, transpose_b=True)
            # Qinv = Kuu⁻¹ - L⁻T S L⁻¹
            # Linv = (L⁻¹ I) = solve(L, I)
            # Kinv = Linv.T @ Linv
        I = tf.eye(tf.shape(Linv_cov_u_LinvT)[-1], dtype=Linv_cov_u_LinvT.dtype)
        B = I - Linv_cov_u_LinvT
        LinvT_B = tf.linalg.triangular_solve(L, B, adjoint=True)
        B_Linv = tf.linalg.adjoint(LinvT_B)
        Qinv = tf.linalg.triangular_solve(L, B_Linv, adjoint=True)

        return alpha, Qinv

    def update_cache(self):
        self.alpha, self.Qinv = self._precompute()

    def freeze(self):
        self.alpha = Parameter(alpha, trainable=False)
        self.Qinv = Parameter(Qinv, trainable=False)

    def update_cache_with_variables(self):
        alpha, Qinv = self._precompute()
        if isinstance(self.alpha, Parameter) and isinstance(self.Qinv, Parameter):
            self.alpha.assign(alpha)
            self.Qinv.assign(Qinv)
        else:
            self.alpha = Parameter(alpha, trainable=False)
            self.Qinv = Parameter(Qinv, trainable=False)

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        # Qinv: [L, M, M]
        # alpha: [M, L]

        # Kuf = covariances.Kuf(self.iv, self.kernel, Xnew)  # [(R), M, N]
        # mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        # if Kuf.shape.ndims == 3:
        #     mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        # if isinstance(self.kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):

        #     Knn = tf.stack([k(Xnew, full_cov=full_cov) for k in self.kernel.kernels], axis=0)
        # elif isinstance(self.kernel, kernels.MultioutputKernel):
        #     Knn = self.kernel.kernel(Xnew, full_cov=full_cov)
        # else:
        #     Knn = self.kernel(Xnew, full_cov=full_cov)
        Kuf, Knn = _get_kernels(Xnew, self.iv, self.kernel, full_cov, full_output_cov) 

        mean = tf.matmul(Kuf, self.alpha, transpose_a=True)
        if Kuf.shape.ndims == 3:
            mean = tf.linalg.adjoint(tf.squeeze(mean, axis=-1))

        if full_cov:
            Kfu_Qinv_Kuf = tf.matmul(Kuf, tf.matmul(self.Qinv, Kuf), transpose_a=True)
            cov = Knn - Kfu_Qinv_Kuf
        else:
            # [AT B]_ij = AT_ik B_kj = A_ki B_kj
            # TODO check whether einsum is faster now?
            Kfu_Qinv_Kuf = tf.reduce_sum(Kuf * tf.matmul(self.Qinv, Kuf), axis=-2)
            cov = Knn - Kfu_Qinv_Kuf
            cov = tf.linalg.adjoint(cov)

        if isinstance(self.kernel, kernels.LinearCoregionalization):
            cov = expand_independent_outputs(cov, full_cov, full_output_cov=False)
            mean, cov = mix_latent_gp(self.kernel.W, mean, cov, full_cov, full_output_cov)
        else:
            cov = expand_independent_outputs(cov, full_cov, full_output_cov)

        return mean + self.mean_function(Xnew), cov


def _get_kernels(Xnew, iv, kernel, full_cov, full_output_cov):

    Kuf = covariances.Kuf(iv, kernel, Xnew)  # [(R), M, N]

    if isinstance(kernel, (kernels.SeparateIndependent, kernels.IndependentLatent)):

        Knn = tf.stack([k(Xnew, full_cov=full_cov) for k in kernel.kernels], axis=0)
    elif isinstance(kernel, kernels.MultioutputKernel):
        Knn = kernel.kernel(Xnew, full_cov=full_cov)
    else:
        Knn = kernel(Xnew, full_cov=full_cov)

    return Kuf, Knn
