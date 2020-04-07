# Copyright 2016 Valentine Svensson, James Hensman, alexggmatthews, Alexis Boukouvalas
# Copyright 2017 Artem Artemev @awav
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
Likelihoods are another core component of GPflow. This describes how likely the
data is under the assumptions made about the underlying latent functions
p(Y|F). Different likelihoods make different
assumptions about the distribution of the data, as such different data-types
(continuous, binary, ordinal, count) are better modelled with different
likelihood assumptions.

Use of any likelihood other than Gaussian typically introduces the need to use
an approximation to perform inference, if one isn't already needed. A
variational inference and MCMC models are included in GPflow and allow
approximate inference with non-Gaussian likelihoods. An introduction to these
models can be found :ref:`here <implemented_models>`. Specific notebooks
illustrating non-Gaussian likelihood regressions are available for
`classification <notebooks/classification.html>`_ (binary data), `ordinal
<notebooks/ordinal.html>`_ and `multiclass <notebooks/multiclass.html>`_.

Creating new likelihoods
----------
Likelihoods are defined by their
log-likelihood. When creating new likelihoods, the
:func:`logp <gpflow.likelihoods.Likelihood.logp>` method (log p(Y|F)), the
:func:`conditional_mean <gpflow.likelihoods.Likelihood.conditional_mean>`,
:func:`conditional_variance
<gpflow.likelihoods.Likelihood.conditional_variance>`.

In order to perform variational inference with non-Gaussian likelihoods a term
called ``variational expectations``, ∫ q(F) log p(Y|F) dF, needs to
be computed under a Gaussian distribution q(F) ~ N(μ, Σ).

The :func:`variational_expectations <gpflow.likelihoods.Likelihood.variational_expectations>`
method can be overriden if this can be computed in closed form, otherwise; if
the new likelihood inherits
:class:`Likelihood <gpflow.likelihoods.Likelihood>` the default will use
Gauss-Hermite numerical integration (works well when F is 1D
or 2D), if the new likelihood inherits from
:class:`MonteCarloLikelihood <gpflow.likelihoods.MonteCarloLikelihood>` the
integration is done by sampling (can be more suitable when F is higher dimensional).
"""

import numpy as np
import tensorflow as tf
import abc
import warnings

from ..base import Module
from ..quadrature import hermgauss, ndiag_mc, ndiagquad


class Likelihood(Module, metaclass=abc.ABCMeta):
    def __init__(self, latent_dim: int, observation_dim: int):
        """
        A base class for likelihoods, which specifies an observation model 
        connecting the latent functions ('F') to the data ('Y').

        All of the members of this class are expected to obey some shape conventions, as specified
        by latent_dim and observation_dim.

        If we're operating on an array of function values 'F', then the last dimension represents
        multiple functions (preceding dimensions could represent different data points, or
        different random samples, for example). Similarly, the last dimension of Y represents a
        single data point. We check that the dimensions are as this object expects.

        The return shapes of all functions in this class is the broadcasted shape of the arguments,
        excluding the last dimension of each argument.

        :param latent_dim: the dimension of the vector F of latent functions for a single data point
        :param observation_dim: the dimension of the observation vector Y for a single data point
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_dim = observation_dim

    def _check_last_dims_valid(self, F, Y):
        """
        Assert that the dimensions of the latent functions F and the data Y are compatible.

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]
        """
        self._check_latent_dims(F)
        self._check_data_dims(Y)

    def _check_return_shape(self, result, F, Y):
        """
        Check that the shape of a computed statistic of the data
        is the broadcasted shape from F and Y.

        :param result: result Tensor, with shape [...]
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]
        """
        expected_shape = tf.broadcast_dynamic_shape(tf.shape(F)[:-1], tf.shape(Y)[:-1])
        tf.debugging.assert_equal(tf.shape(result), expected_shape)

    def _check_latent_dims(self, F):
        """
        Ensure that a tensor of latent functions F has latent_dim as right-most dimension.

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        """
        tf.debugging.assert_shapes([(F, (..., self.latent_dim))])

    def _check_data_dims(self, Y):
        """
        Ensure that a tensor of data Y has observation_dim as right-most dimension.

        :param Y: observation Tensor, with shape [..., observation_dim]
        """
        tf.debugging.assert_shapes([(Y, (..., self.observation_dim))])

    def log_prob(self, F, Y):
        """
        The log probability density log p(Y|F)

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log pdf, with shape [...]
        """
        self._check_last_dims_valid(F, Y)
        res = self._log_prob(F, Y)
        self._check_return_shape(res, F, Y)
        return res

    @abc.abstractmethod
    def _log_prob(self, F, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        """
        The conditional mean of Y|F: [E[Y₁|F], ..., E[Yₖ|F]]
        where K = observation_dim

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean [..., observation_dim]
        """
        self._check_latent_dims(F)
        expected_Y = self._conditional_mean(F)
        self._check_data_dims(expected_Y)
        return expected_Y

    def _conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        """
        The conditional marginal variance of Y|F: [var(Y₁|F), ..., var(Yₖ|F)]
        where K = observation_dim

        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :returns: variance [..., observation_dim]
        """
        self._check_latent_dims(F)
        var_Y = self._conditional_variance(F)
        self._check_data_dims(var_Y)
        return var_Y

    def _conditional_variance(self, F):
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
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


        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance, both with shape [..., observation_dim]
        """
        self._check_latent_dims(Fmu)
        self._check_latent_dims(Fvar)
        mu, var = self._predict_mean_and_var(Fmu, Fvar)
        self._check_data_dims(mu)
        self._check_data_dims(var)
        return mu, var

    @abc.abstractmethod
    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def predict_log_density(self, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y,

        i.e. if
            q(F) = N(Fmu, Fvar)

        and this object represents

            p(y|F)

        then this method computes the predictive density

            log ∫ p(y=Y|F)q(F) df

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log predictive density, with shape [...]
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        self._check_last_dims_valid(Fmu, Y)
        res = self._predict_log_density(Fmu, Fvar, Y)
        self._check_return_shape(res, Fmu, Y)
        return res

    @abc.abstractmethod
    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        """
        Deprecated: see `predict_log_density`
        """
        warnings.warn(
            "predict_density is deprecated and will be removed in GPflow 2.1, use predict_log_density instead",
            DeprecationWarning,
        )
        return self.predict_log_density(Fmu, Fvar, Y)

    def variational_expectations(self, Fmu, Fvar, Y):
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
        ret = self._variational_expectations(Fmu, Fvar, Y)
        self._check_return_shape(ret, Fmu, Y)
        return ret

    @abc.abstractmethod
    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError


class ScalarLikelihood(Likelihood):
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
        self.num_gauss_hermite_points = 20

    def _check_last_dims_valid(self, F, Y):
        """
        Assert that the dimensions of the latent functions and the data are compatible
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        tf.debugging.assert_shapes([(F, (..., "num_latent")), (Y, (..., "num_latent"))])

    def _log_prob(self, F, Y):
        r"""
        Compute log p(Y|F), where by convention we sum out the last axis as it represented
        independent latent functions and observations.
        :param F: function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]
        """
        return tf.reduce_sum(self._scalar_log_prob(F, Y), axis=-1)

    @abc.abstractmethod
    def _scalar_log_prob(self, F, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]:
        :returns: variational expectations, with shape [...]
        """
        return tf.reduce_sum(
            ndiagquad(self._scalar_log_prob, self.num_gauss_hermite_points, Fmu, Fvar, Y=Y),
            axis=-1,
        )

    def _predict_log_density(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]:
        :returns: log predictive density, with shape [...]
        """
        return tf.reduce_sum(
            ndiagquad(
                self._scalar_log_prob, self.num_gauss_hermite_points, Fmu, Fvar, logspace=True, Y=Y,
            ),
            axis=-1,
        )

    def _predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.

        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance, both with shape [..., observation_dim]
        """

        def integrand(*X):
            return self.conditional_variance(*X) + self.conditional_mean(*X) ** 2

        integrands = [self.conditional_mean, integrand]
        E_y, E_y2 = ndiagquad(integrands, self.num_gauss_hermite_points, Fmu, Fvar)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y


class SwitchedLikelihood(ScalarLikelihood):
    def __init__(self, likelihood_list, **kwargs):
        """
        In this likelihood, we assume at extra column of Y, which contains
        integers that specify a likelihood from the list of likelihoods.
        """
        super().__init__(**kwargs)
        for l in likelihood_list:
            assert isinstance(l, ScalarLikelihood)
        self.likelihoods = likelihood_list

    def _partition_and_stitch(self, args, func_name):
        """
        args is a list of tensors, to be passed to self.likelihoods.<func_name>

        args[-1] is the 'Y' argument, which contains the indexes to self.likelihoods.

        This function splits up the args using dynamic_partition, calls the
        relevant function on the likelihoods, and re-combines the result.
        """
        # get the index from Y
        Y = args[-1]
        ind = Y[..., -1]
        ind = tf.cast(ind, tf.int32)
        Y = Y[..., :-1]
        args[-1] = Y

        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = zip(*[tf.dynamic_partition(X, ind, len(self.likelihoods)) for X in args])

        # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i) for f, args_i in zip(funcs, args)]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, len(self.likelihoods))
        results = tf.dynamic_stitch(partitions, results)

        return results

    def _check_last_dims_valid(self, F, Y):
        tf.assert_equal(tf.shape(F)[-1], tf.shape(Y)[-1] - 1)

    def _scalar_log_prob(self, F, Y):
        return self._partition_and_stitch([F, Y], "_scalar_log_prob")

    def _predict_log_density(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "predict_log_density")

    def _variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")

    def _predict_mean_and_var(self, Fmu, Fvar):
        mvs = [lik.predict_mean_and_var(Fmu, Fvar) for lik in self.likelihoods]
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, 1)
        var = tf.concat(var_list, 1)
        return mu, var

    def _conditional_mean(self, F):
        raise NotImplementedError

    def _conditional_variance(self, F):
        raise NotImplementedError


class MonteCarloLikelihood(Likelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_monte_carlo_points = 100

    def _mc_quadrature(self, funcs, Fmu, Fvar, logspace: bool = False, epsilon=None, **Ys):
        return ndiag_mc(funcs, self.num_monte_carlo_points, Fmu, Fvar, logspace, epsilon, **Ys)

    def _predict_mean_and_var(self, Fmu, Fvar, epsilon=None):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           ∫∫ y p(y|f)q(f) df dy

        and the predictive variance

           ∫∫ y² p(y|f)q(f) df dy  - [ ∫∫ y p(y|f)q(f) df dy ]²

        Here, we implement a default Monte Carlo routine.
        """
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = self._mc_quadrature(
            [self.conditional_mean, integrand2], Fmu, Fvar, epsilon=epsilon
        )
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y  # [N, D]

    def _predict_log_density(self, Fmu, Fvar, Y, epsilon=None):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            log ∫ p(y=Y|f)q(f) df

        Here, we implement a default Monte Carlo routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, logspace=True, epsilon=epsilon),
            axis=-1,
        )

    def _variational_expectations(self, Fmu, Fvar, Y, epsilon=None):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)  - Fmu: [N, D]  Fvar: [N, D]

        and this object represents

            p(y|f)  - Y: [N, 1]

        then this method computes

           ∫ (log p(y|f)) q(f) df.


        Here, we implement a default Monte Carlo quadrature routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, epsilon=epsilon), axis=-1
        )
