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

from typing import Optional

import numpy as np
import tensorflow as tf

from .. import covariances, kernels, likelihoods
from ..base import InputData, MeanAndVariance, OutputData, Parameter, RegressionData, TensorType
from ..config import default_float, default_jitter
from ..expectations import expectation
from ..experimental.check_shapes import check_shapes, inherit_check_shapes
from ..inducing_variables import InducingPoints
from ..kernels import Kernel
from ..mean_functions import MeanFunction, Zero
from ..probability_distributions import DiagonalGaussian
from ..utilities import assert_params_false, positive, to_default_float
from ..utilities.ops import pca_reduce
from .gpr import GPR
from .model import GPModel
from .training_mixins import InternalDataTrainingLossMixin
from .util import InducingVariablesLike, data_input_to_tensor, inducingpoint_wrapper


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    @check_shapes(
        "data: [N, P]",
        "X_data_mean: [N, Q]",
    )
    def __init__(
        self,
        data: OutputData,
        latent_dim: int,
        X_data_mean: Optional[tf.Tensor] = None,
        kernel: Optional[Kernel] = None,
        mean_function: Optional[MeanFunction] = None,
    ):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.

        :param data: y data matrix.
        :param latent_dim: the number of latent dimensions (Q)
        :param X_data_mean: latent positions, for the initialisation of the latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param mean_function: mean function, by default None.
        """
        if X_data_mean is None:
            X_data_mean = pca_reduce(data, latent_dim)

        num_latent_gps = X_data_mean.shape[1]
        if num_latent_gps != latent_dim:
            msg = "Passed in number of latent {0} does not match initial X {1}."
            raise ValueError(msg.format(latent_dim, num_latent_gps))

        if mean_function is None:
            mean_function = Zero()

        if kernel is None:
            kernel = kernels.SquaredExponential(lengthscales=tf.ones((latent_dim,)))

        if data.shape[1] < num_latent_gps:
            raise ValueError("More latent dimensions than observed.")

        gpr_data = (Parameter(X_data_mean), data_input_to_tensor(data))
        super().__init__(gpr_data, kernel, mean_function=mean_function)


class BayesianGPLVM(GPModel, InternalDataTrainingLossMixin):
    @check_shapes(
        "data: [N, P]",
        "X_data_mean: [N, Q]",
        "X_data_var: [N, Q]",
        "X_prior_mean: [N, Q]",
        "X_prior_var: [N, Q]",
    )
    def __init__(
        self,
        data: OutputData,
        X_data_mean: tf.Tensor,
        X_data_var: tf.Tensor,
        kernel: Kernel,
        num_inducing_variables: Optional[int] = None,
        inducing_variable: Optional[InducingVariablesLike] = None,
        X_prior_mean: Optional[TensorType] = None,
        X_prior_var: Optional[TensorType] = None,
    ):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.

        :param data: data matrix, size N (number of points) x P (dimensions)
        :param X_data_mean: initial latent positions, size N (number of points) x
            Q (latent dimensions).
        :param X_data_var: variance of latent positions ([N, Q]), for the initialisation of the
            latent space.
        :param kernel: kernel specification, by default Squared Exponential
        :param num_inducing_variables: number of inducing points, M
        :param inducing_variable: matrix of inducing points, size M (inducing points) x
            Q (latent dimensions). By default random permutation of X_data_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0.
            Same size as X_data_mean.
        :param X_prior_var: prior variance used in KL term of bound. By default 1.
        """
        num_data, num_latent_gps = X_data_mean.shape
        super().__init__(kernel, likelihoods.Gaussian(), num_latent_gps=num_latent_gps)
        self.data = data_input_to_tensor(data)

        self.X_data_mean = Parameter(X_data_mean)
        self.X_data_var = Parameter(X_data_var, transform=positive())

        self.num_data = num_data
        self.output_dim = self.data.shape[-1]

        if (inducing_variable is None) == (num_inducing_variables is None):
            raise ValueError(
                "BayesianGPLVM needs exactly one of `inducing_variable` and"
                " `num_inducing_variables`"
            )

        if inducing_variable is None:
            # By default we initialize by subset of initial latent points
            # Note that tf.random.shuffle returns a copy, it does not shuffle in-place
            Z = tf.random.shuffle(X_data_mean)[:num_inducing_variables]
            inducing_variable = InducingPoints(Z)

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        assert X_data_mean.shape[1] == self.num_latent_gps

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = tf.zeros((self.num_data, self.num_latent_gps), dtype=default_float())
        if X_prior_var is None:
            X_prior_var = tf.ones((self.num_data, self.num_latent_gps))

        self.X_prior_mean = tf.convert_to_tensor(np.atleast_1d(X_prior_mean), dtype=default_float())
        self.X_prior_var = tf.convert_to_tensor(np.atleast_1d(X_prior_var), dtype=default_float())

    # type-ignore is because of changed method signature:
    @inherit_check_shapes
    def maximum_log_likelihood_objective(self) -> tf.Tensor:  # type: ignore[override]
        return self.elbo()

    @check_shapes(
        "return: []",
    )
    def elbo(self) -> tf.Tensor:
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """
        Y_data = self.data

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        num_inducing = self.inducing_variable.num_inducing
        psi0 = tf.reduce_sum(expectation(pX, self.kernel))
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        cov_uu = covariances.Kuu(self.inducing_variable, self.kernel, jitter=default_jitter())
        L = tf.linalg.cholesky(cov_uu)
        sigma2 = self.likelihood.variance

        # Compute intermediate matrices
        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        log_det_B = 2.0 * tf.reduce_sum(tf.math.log(tf.linalg.diag_part(LB)))
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2

        # KL[q(x) || p(x)]
        dX_data_var = (
            self.X_data_var
            if self.X_data_var.shape.ndims == 2
            else tf.linalg.diag_part(self.X_data_var)
        )
        NQ = to_default_float(tf.size(self.X_data_mean))
        D = to_default_float(tf.shape(Y_data)[1])
        KL = -0.5 * tf.reduce_sum(tf.math.log(dX_data_var))
        KL += 0.5 * tf.reduce_sum(tf.math.log(self.X_prior_var))
        KL -= 0.5 * NQ
        KL += 0.5 * tf.reduce_sum(
            (tf.square(self.X_data_mean - self.X_prior_mean) + dX_data_var) / self.X_prior_var
        )

        # compute log marginal bound
        ND = to_default_float(tf.size(Y_data))
        bound = -0.5 * ND * tf.math.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y_data)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 - tf.reduce_sum(tf.linalg.diag_part(AAT)))
        bound -= KL
        return bound

    @inherit_check_shapes
    def predict_f(
        self, Xnew: InputData, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.

        Note: This model does not allow full output covariances.

        :param Xnew: points at which to predict
        """
        assert_params_false(self.predict_f, full_output_cov=full_output_cov)

        pX = DiagonalGaussian(self.X_data_mean, self.X_data_var)

        Y_data = self.data
        num_inducing = self.inducing_variable.num_inducing
        psi1 = expectation(pX, (self.kernel, self.inducing_variable))
        psi2 = tf.reduce_sum(
            expectation(
                pX, (self.kernel, self.inducing_variable), (self.kernel, self.inducing_variable)
            ),
            axis=0,
        )
        jitter = default_jitter()
        Kus = covariances.Kuf(self.inducing_variable, self.kernel, Xnew)
        sigma2 = self.likelihood.variance
        L = tf.linalg.cholesky(covariances.Kuu(self.inducing_variable, self.kernel, jitter=jitter))

        A = tf.linalg.triangular_solve(L, tf.transpose(psi1), lower=True)
        tmp = tf.linalg.triangular_solve(L, psi2, lower=True)
        AAT = tf.linalg.triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + tf.eye(num_inducing, dtype=default_float())
        LB = tf.linalg.cholesky(B)
        c = tf.linalg.triangular_solve(LB, tf.linalg.matmul(A, Y_data), lower=True) / sigma2
        tmp1 = tf.linalg.triangular_solve(L, Kus, lower=True)
        tmp2 = tf.linalg.triangular_solve(LB, tmp1, lower=True)
        mean = tf.linalg.matmul(tmp2, c, transpose_a=True)
        if full_cov:
            var = (
                self.kernel(Xnew)
                + tf.linalg.matmul(tmp2, tmp2, transpose_a=True)
                - tf.linalg.matmul(tmp1, tmp1, transpose_a=True)
            )
            shape = tf.stack([tf.shape(Y_data)[1], 1, 1])
            var = tf.tile(tf.expand_dims(var, 0), shape)
        else:
            var = (
                self.kernel(Xnew, full_cov=False)
                + tf.reduce_sum(tf.square(tmp2), axis=0)
                - tf.reduce_sum(tf.square(tmp1), axis=0)
            )
            shape = tf.stack([1, tf.shape(Y_data)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    @inherit_check_shapes
    def predict_log_density(
        self, data: RegressionData, full_cov: bool = False, full_output_cov: bool = False
    ) -> tf.Tensor:
        raise NotImplementedError
