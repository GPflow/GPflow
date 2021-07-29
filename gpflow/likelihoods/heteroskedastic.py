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

"""
Likelihoods which require the input location X in addition to F and Y
"""

import abc
import warnings
from typing import Optional

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from . import Likelihood
from .base import DEFAULT_NUM_GAUSS_HERMITE_POINTS
from .. import logdensities
from ..base import Parameter
from ..quadrature import GaussianQuadrature, NDiagGHQuadrature
from ..utilities import positive


class HetereoskedasticLikelihood(Likelihood):


    def conditional_variance(self, X, F):
        """
        The conditional marginal variance of Y|F: [var(Y₁|F), ..., var(Yₖ|F)]
        where K = observation_dim

        :param X: input location Tensor, with shape [..., inputs_dim]
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: variance [..., observation_dim]
        """
        self._check_latent_dims(F)
        var_Y = self._conditional_variance(X, F)
        self._check_data_dims(var_Y)
        return var_Y

    def _conditional_variance(self, X, F):
        raise NotImplementedError

    def log_prob(self, X, F, Y):
        """
        The log probability density log p(Y|F)

        :param X: input locations Tensor, with shape [..., input_dim]:
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log pdf, with shape [...]
        """
        self._check_last_dims_valid(F, Y)
        res = self._log_prob(X, F, Y)
        self._check_return_shape(res, F, Y)
        return res

    @abc.abstractmethod
    def _log_prob(self, X, F, Y):
        raise NotImplementedError

    def predict_mean_and_var(self, X, Fmu, Fvar):
        """
        Given a Normal distribution for the latent function,
        return the mean and marginal variance of Y,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           ∫∫ y p(y|f)q(f) df dy

        and the predictive variance

           ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²


        :param X: input locations Tensor, with shape [..., input_dim]:
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance, both with shape [..., observation_dim]
        """
        self._check_latent_dims(Fmu)
        self._check_latent_dims(Fvar)
        mu, var = self._predict_mean_and_var(X, Fmu, Fvar)
        self._check_data_dims(mu)
        self._check_data_dims(var)
        return mu, var

    @abc.abstractmethod
    def _predict_mean_and_var(self, X, Fmu, Fvar):
        raise NotImplementedError

    def predict_log_density(self, X, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y,

        i.e. if
            q(F) = N(Fmu, Fvar)

        and this object represents

            p(y|F)

        then this method computes the predictive density

            log ∫ p(y=Y|F)q(F) df

        :param X: input locations Tensor, with shape [..., input_dim]:
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log predictive density, with shape [...]
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        self._check_last_dims_valid(Fmu, Y)
        res = self._predict_log_density(X, Fmu, Fvar, Y)
        self._check_return_shape(res, Fmu, Y)
        return res

    @abc.abstractmethod
    def _predict_log_density(self, X, Fmu, Fvar, Y):
        raise NotImplementedError

    def predict_density(self, X, Fmu, Fvar, Y):
        """
        Deprecated: see `predict_log_density`
        """
        warnings.warn(
            "predict_density is deprecated and will be removed in GPflow 2.1, use predict_log_density instead",
            DeprecationWarning,
        )
        return self.predict_log_density(X, Fmu, Fvar, Y)

    def variational_expectations(self, X, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values,

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           ∫ log(p(y=Y|f)) q(f) df.

        This only works if the broadcasting dimension of the statistics of q(f) (mean and variance)
        are broadcastable with that of the data Y.

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: expected log density of the data given q(F), with shape [...]
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        # returns an error if Y[:-1] and Fmu[:-1] do not broadcast together
        _ = tf.broadcast_dynamic_shape(tf.shape(Fmu)[:-1], tf.shape(Y)[:-1])
        self._check_last_dims_valid(Fmu, Y)
        ret = self._variational_expectations(X, Fmu, Fvar, Y)
        self._check_return_shape(ret, Fmu, Y)
        return ret

    @abc.abstractmethod
    def _variational_expectations(self, X, Fmu, Fvar, Y):
        raise NotImplementedError


class HetQuadratureLikelihood(HetereoskedasticLikelihood):
    def __init__(
        self,
        latent_dim: int,
        observation_dim: int,
        *,
        quadrature: Optional[GaussianQuadrature] = None,
    ):
        super().__init__(latent_dim=latent_dim, observation_dim=observation_dim)
        if quadrature is None:
            with tf.init_scope():
                quadrature = NDiagGHQuadrature(
                    self._quadrature_dim, DEFAULT_NUM_GAUSS_HERMITE_POINTS
                )
        self.quadrature = quadrature

    @property
    def _quadrature_dim(self) -> int:
        """
        This defines the number of dimensions over which to evaluate the
        quadrature. Generally, this is equal to self.latent_dim. This exists
        as a separate property to allow the ScalarLikelihood subclass to
        override it with 1 (broadcasting over observation/latent dimensions
        instead).
        """
        return self.latent_dim

    def _quadrature_log_prob(self, X, F, Y):
        """
        Returns the appropriate log prob integrand for quadrature.

        Quadrature expects f(X), here logp(F), to return shape [N_quad_points]
        + batch_shape + [d']. Here d'=1, but log_prob() only returns
        [N_quad_points] + batch_shape, so we add an extra dimension.

        Also see _quadrature_reduction.
        """
        return tf.expand_dims(self.log_prob(X, F, Y), axis=-1)

    def _quadrature_reduction(self, quadrature_result):
        """
        Converts the quadrature integral appropriately.

        The return shape of quadrature is batch_shape + [d']. Here, d'=1, but
        we want predict_log_density and variational_expectations to return just
        batch_shape, so we squeeze out the extra dimension.

        Also see _quadrature_log_prob.
        """
        return tf.squeeze(quadrature_result, axis=-1)

    def _predict_log_density(self, X, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log predictive density, with shape [...]
        """
        return self._quadrature_reduction(
            self.quadrature.logspace(self._quadrature_log_prob, Fmu, Fvar, Y=Y)
        )

    def _variational_expectations(self, X, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: variational expectations, with shape [...]
        """
        return self._quadrature_reduction(
            self.quadrature(self._quadrature_log_prob, Fmu, Fvar, Y=Y)
        )

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance of Y, both with shape [..., observation_dim]
        """

        def conditional_y_squared(*F):
            return self.conditional_variance(*F) + tf.square(self.conditional_mean(*F))

        E_y, E_y2 = self.quadrature([self.conditional_mean, conditional_y_squared], Fmu, Fvar)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y


class HetScalarLikelihood(HetQuadratureLikelihood):
    """
    A likelihood class that helps with scalar likelihood functions: likelihoods where
    each scalar latent function is associated with a single scalar observation variable.

    If there are multiple latent functions, then there must be a corresponding number of data: we
    check for this.

    The `Likelihood` class contains methods to compute marginal statistics of functions
    of the latents and the data ϕ(y,f):
     * variational_expectations:  ϕ(y,f) = log p(y|f)
     * predict_log_density: ϕ(y,f) = p(y|f)
    Those statistics are computed after having first marginalized the latent processes f
    under a multivariate normal distribution q(f) that is fully factorized.

    Some univariate integrals can be done by quadrature: we implement quadrature routines for 1D
    integrals in this class, though they may be overwritten by inheriting classes where those
    integrals are available in closed form.
    """

    def __init__(self, **kwargs):
        super().__init__(latent_dim=None, observation_dim=None, **kwargs)

    @property
    def num_gauss_hermite_points(self) -> int:
        warnings.warn(
            "The num_gauss_hermite_points property is deprecated; access through the `quadrature` attribute instead",
            DeprecationWarning,
        )

        if not isinstance(self.quadrature, NDiagGHQuadrature):
            raise TypeError(
                "Can only query num_gauss_hermite_points if quadrature is a NDiagGHQuadrature instance"
            )
        return self.quadrature.n_gh

    @num_gauss_hermite_points.setter
    def num_gauss_hermite_points(self, n_gh: int):
        warnings.warn(
            "The num_gauss_hermite_points setter is deprecated; assign a new GaussianQuadrature instance to the `quadrature` attribute instead",
            DeprecationWarning,
        )

        if isinstance(self.quadrature, NDiagGHQuadrature) and n_gh == self.quadrature.n_gh:
            return  # nothing to do here

        with tf.init_scope():
            self.quadrature = NDiagGHQuadrature(self._quadrature_dim, n_gh)

    def _check_last_dims_valid(self, F, Y):
        """
        Assert that the dimensions of the latent functions and the data are compatible
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        tf.debugging.assert_shapes([(F, (..., "num_latent")), (Y, (..., "num_latent"))])

    def _log_prob(self, X, F, Y):
        r"""
        Compute log p(Y|F), where by convention we sum out the last axis as it represented
        independent latent functions and observations.
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        return tf.reduce_sum(self._scalar_log_prob(X, F, Y), axis=-1)

    @abc.abstractmethod
    def _scalar_log_prob(self, X, F, Y):
        raise NotImplementedError

    @property
    def _quadrature_dim(self) -> int:
        """
        Quadrature is over the latent dimensions. Generally, this is equal to
        self.latent_dim. This separate property allows the ScalarLikelihood
        subclass to override it with 1 (broadcasting over observation/latent
        dimensions instead).
        """
        return 1

    def _quadrature_log_prob(self, X, F, Y):
        """
        Returns the appropriate log prob integrand for quadrature.

        Quadrature expects f(X), here logp(F), to return shape [N_quad_points]
        + batch_shape + [d']. Here d' corresponds to the last dimension of both
        F and Y, and _scalar_log_prob simply broadcasts over this.

        Also see _quadrature_reduction.
        """
        return self._scalar_log_prob(X, F, Y)

    def _quadrature_reduction(self, quadrature_result):
        """
        Converts the quadrature integral appropriately.

        The return shape of quadrature is batch_shape + [d']. Here, d'
        corresponds to the last dimension of both F and Y, and we want to sum
        over the observations to obtain the overall predict_log_density or
        variational_expectations.

        Also see _quadrature_log_prob.
        """
        return tf.reduce_sum(quadrature_result, axis=-1)


class HeteroskedasticGaussianLikelihood(HetScalarLikelihood):
    r"""
    The HeteroskedasticGaussian likelihood is appropriate where uncertainties associated with
    the data are believed to follow a normal distribution, with a variance which potentially evolves with input location.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the
    likelihood variance by default.
    """

    DEFAULT_VARIANCE_LOWER_BOUND = 1e-6

    def __init__(self, variance=1.0, ndims: int=1, variance_lower_bound=DEFAULT_VARIANCE_LOWER_BOUND, **kwargs):
        """
        :param variance: The noise variance; must be greater than
            ``variance_lower_bound``.
        :param variance_lower_bound: The lower (exclusive) bound of ``variance``.
        :param kwargs: Keyword arguments forwarded to :class:`ScalarLikelihood`.
        """
        super().__init__(**kwargs)

        if variance <= variance_lower_bound:
            raise ValueError(
                f"The variance of the Gaussian likelihood must be strictly greater than {variance_lower_bound}"
            )

        shift_prior = tfp.distributions.Cauchy(loc=np.float64(0.0), scale=np.float64(5.0))
        base_prior = tfp.distributions.LogNormal(loc=np.float64(-2.0), scale=np.float64(2.0))
        self.variance = Parameter(np.ones(ndims), transform=positive(lower=variance_lower_bound))
        self.shifts = Parameter(np.zeros(ndims), trainable=True, prior=shift_prior, name="shifts")
        self.likelihood_variance = Parameter(0.1, transform=positive(lower=variance_lower_bound), prior=base_prior)

    def compute_variances(self, X):
        """ Determine the likelihood variance at the specified input locations X. """

        Z = X + self.shifts
        normalised_variance = self.variance / (1 + self.shifts ** 2)
        het_variance = tf.reduce_sum(tf.square(Z) * normalised_variance, axis=-1, keepdims=True)
        return het_variance + self.likelihood_variance

    def _scalar_log_prob(self, X, F, Y):
        variances = self.compute_variances(X)
        return logdensities.gaussian(Y, F, variances)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, X, F):
        variances = self.compute_variances(X)
        return tf.fill(tf.shape(F), tf.squeeze(variances))

    def _predict_mean_and_var(self, X, Fmu, Fvar):
        variances = self.compute_variances(X)
        return tf.identity(Fmu), Fvar + variances

    def _predict_log_density(self, X, Fmu, Fvar, Y):
        variances = self.compute_variances(X)
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + variances), axis=-1)

    def _variational_expectations(self, X, Fmu, Fvar, Y):
        variances = self.compute_variances(X)
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(variances)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / variances,
            axis=-1,
        )
