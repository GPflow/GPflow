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

from .. import logdensities
from ..base import Module, Parameter
from ..config import default_float, default_int
from ..quadrature import hermgauss, ndiag_mc, ndiagquad
from ..utilities import positive
from .robustmax import RobustMax


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 - 2 * jitter) + jitter


class Likelihood(Module):
    def __init__(self, num_latent_functions: int, num_data_dims: int):
        """
        A base class for likelihoods, which specifies an observation model 
        connecting the latent functions ('F') to the data ('Y').

        All of the members of this class are expected to obey some shape conventions, as specified
        by num_latent_functions and num_data_dims.

        If we're operating on an array of function values 'F', then the last dimension represents
        multiple functions (preceeding dimensions could represent different data points, or
        different random samples, for example). Similarly, the last dimension of Y represents a
        single data point. We check that the dimensions are as this object expects.

        The return shapes of all functions in this class is the broadcasted shape of the arguments,
        up to the last dimensions.
        """
        super().__init__()
        self.num_gauss_hermite_points = 20
        self.num_latent_functions = num_latent_functions
        self.num_data_dims = num_data_dims

    def check_last_dims_match(self, F, Y):
        """
        Assert that the dimensions of the latent functions and the data are compatible
        """
        self.check_latent_dims(F)
        self.check_data_dims(Y)

    def check_return_shape(self, result, F, Y):
        """
        Check that the shape of a computed result is the broadcasted shape from F and Y
        """
        expected_shape = tf.broadcast_dynamic_shape(tf.shape(F)[:-1], tf.shape(Y)[:-1])
        tf.debugging.assert_equal(tf.shape(result), expected_shape)

    def check_latent_dims(self, F):
        """
        ensure that a tensor of latent functions has
        num_latent_functions as right most dimension 
        """
        tf.debugging.assert_shapes([(F, (..., self.num_latent_functions))])

    def check_data_dims(self, Y):
        """
        ensure that a tensor of data has num_data_dims as right most dimension.
        """
        tf.debugging.assert_shapes([(Y, (..., self.num_data_dims))])

    def log_prob(self, F, Y):
        self.check_last_dims_match(F, Y)
        ret = self._log_prob(F, Y)
        self.check_return_shape(ret, F, Y)
        return ret

    def _log_prob(self, F, Y):
        raise NotImplementedError

    def conditional_mean(self, F):
        self.check_latent_dims(F)
        expected_Y = self._conditional_mean(F)
        self.check_data_dims(expected_Y)
        return expected_Y

    def _conditional_mean(self, F):
        raise NotImplementedError

    def conditional_variance(self, F):
        self.check_latent_dims(F)
        var_Y = self._conditional_variance(F)
        self.check_data_dims(var_Y)
        return var_Y

    def _conditional_variance(self, F):
        raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        self.check_latent_dims(Fmu)
        self.check_latent_dims(Fvar)
        mu, var = self._predict_mean_and_var(Fmu, Fvar)
        self.check_data_dims(mu)
        self.check_data_dims(var)
        return mu, var

    def _predict_mean_and_var(self, Fmu, Fvar):
        raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        self.check_last_dims_match(Fmu, Y)
        ret = self._predict_density(Fmu, Fvar, Y)
        self.check_return_shape(ret, Fmu, Y)
        return ret

    def _predict_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.
        """
        tf.debugging.assert_equal(tf.shape(Fmu), tf.shape(Fvar))
        self.check_last_dims_match(Fmu, Y)
        ret = self._variational_expectations(Fmu, Fvar, Y)
        self.check_return_shape(ret, Fmu, Y)
        return ret

    def _variational_expectations(self, Fmu, Fvar, Y):
        raise NotImplementedError


class ScalarLikelihood(Likelihood):
    """
    A likelihood class that helps with scalar likelihood functions: likelihoods where each latent
    function is associated with a single observation variable.

    If there are multiple latent functions, then there must be a corresponding number of data: we
    check for this.

    The `Likelihood` class contains methods to compute marginal statistics of the data ϕ(Y),
    where the latent processes F are marginalized under a distribution
    whose pdf q(F) is fully factorized q(F) = 𝚷ₖ q(fₖ).
    
    Some univariate integrals can be done by quadrature: we implement quadrature routines for 1D
    integrals in this class, though they may be overwritten by inherriting classes where those
    integrals are available in closed form.
    """

    def __init__(self, **kwargs):
        super().__init__(num_latent_functions=None, num_data_dims=None, **kwargs)

    def check_last_dims_match(self, F, Y):
        """
        Assert that the dimensions of the latent functions and the data are compatible
        """
        tf.debugging.assert_shapes([(F, (..., "num_latent")), (Y, (..., "num_latent"))])

    def _log_prob(self, F, Y):
        r"""
        Compute log p(Y |F), where by convention we sum out the last axis as it represented
        independent latent functions and observations.
        """
        return tf.reduce_sum(self._scalar_density(F, Y), axis=-1)

    def _scalar_density(self, F, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        nghp = self.num_gauss_hermite_points
        return tf.reduce_sum(ndiagquad(self._scalar_density, nghp, Fmu, Fvar, Y=Y), axis=-1)

    def _predict_density(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        nghp = self.num_gauss_hermite_points
        return tf.reduce_sum(
            ndiagquad(self._scalar_density, nghp, Fmu, Fvar, logspace=True, Y=Y), axis=-1
        )

    def _predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Given a Normal distribution for the latent function,
        return the mean of Y

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive mean

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        """

        def integrand(*X):
            return self.conditional_variance(*X) + self.conditional_mean(*X) ** 2

        integrands = [self.conditional_mean, integrand]
        nghp = self.num_gauss_hermite_points
        E_y, E_y2 = ndiagquad(integrands, nghp, Fmu, Fvar)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y


class Gaussian(ScalarLikelihood):
    r"""
    The Gaussian likelihood is appropriate where uncertainties associated with the data are
    believed to follow a normal distribution, with constant variance.

    Very small uncertainties can lead to numerical instability during the
    optimization process. A lower bound of 1e-6 is therefore imposed on the likelihood variance
    by default.
    """

    def __init__(self, variance=1.0, variance_lower_bound=1e-6, **kwargs):
        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive(lower=variance_lower_bound))

    def _scalar_density(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def _conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def _conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    def _predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def _predict_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(logdensities.gaussian(Y, Fmu, Fvar + self.variance), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        return tf.reduce_sum(
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(self.variance)
            - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance,
            axis=-1,
        )


class Poisson(ScalarLikelihood):
    r"""
    Poisson likelihood for use with count data, where the rate is given by the (transformed) GP.

    let g(.) be the inverse-link function, then this likelihood represents

    p(y_i | f_i) = Poisson(y_i | g(f_i) * binsize)

    Note:binsize
    For use in a Log Gaussian Cox process (doubly stochastic model) where the
    rate function of an inhomogeneous Poisson process is given by a GP.  The
    intractable likelihood can be approximated by gridding the space (into bins
    of size 'binsize') and using this Poisson likelihood.
    """

    def __init__(self, invlink=tf.exp, binsize=1.0, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.binsize = np.array(binsize, dtype=default_float())

    def _scalar_density(self, F, Y):
        return logdensities.poisson(Y, self.invlink(F) * self.binsize)

    def _conditional_variance(self, F):
        return self.invlink(F) * self.binsize

    def _conditional_mean(self, F):
        return self.invlink(F) * self.binsize

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return tf.reduce_sum(
                Y * Fmu
                - tf.exp(Fmu + Fvar / 2) * self.binsize
                - tf.math.lgamma(Y + 1)
                + Y * tf.math.log(self.binsize),
                axis=-1,
            )
        return super()._variational_expectations(Fmu, Fvar, Y)


class Exponential(ScalarLikelihood):
    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def _scalar_density(self, F, Y):
        return logdensities.exponential(Y, self.invlink(F))

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        return tf.square(self.invlink(F))

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return tf.reduce_sum(-tf.exp(-Fmu + Fvar / 2) * Y - Fmu, axis=-1)
        return super()._variational_expectations(Fmu, Fvar, Y)


class StudentT(ScalarLikelihood):
    def __init__(self, scale=1.0, df=3.0, **kwargs):
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale = Parameter(scale, transform=positive(), dtype=default_float())

    def _scalar_density(self, F, Y):
        return logdensities.student_t(Y, F, self.scale, self.df)

    def _conditional_mean(self, F):
        return F

    def _conditional_variance(self, F):
        var = (self.scale ** 2) * (self.df / (self.df - 2.0))
        return tf.fill(tf.shape(F), tf.squeeze(var))


class Bernoulli(ScalarLikelihood):
    def __init__(self, invlink=inv_probit, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def _scalar_density(self, F, Y):
        return logdensities.bernoulli(Y, self.invlink(F))

    def _predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super()._predict_mean_and_var(Fmu, Fvar)

    def _predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return logdensities.bernoulli(Y, p)

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - (p ** 2)


class Gamma(ScalarLikelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """

    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.shape = Parameter(1.0, transform=positive())

    def _scalar_density(self, F, Y):
        return logdensities.gamma(Y, self.shape, self.invlink(F))

    def _conditional_mean(self, F):
        return self.shape * self.invlink(F)

    def _conditional_variance(self, F):
        scale = self.invlink(F)
        return self.shape * (scale ** 2)

    def _variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return tf.reduce_sum(
                -self.shape * Fmu
                - tf.math.lgamma(self.shape)
                + (self.shape - 1.0) * tf.math.log(Y)
                - Y * tf.exp(-Fmu + Fvar / 2.0),
                axis=-1,
            )
        else:
            return super()._variational_expectations(Fmu, Fvar, Y)


class Beta(ScalarLikelihood):
    """
    This uses a reparameterisation of the Beta density. We have the mean of the
    Beta distribution given by the transformed process:

        m = sigma(f)

    and a scale parameter. The familiar alpha, beta parameters are given by

        m     = alpha / (alpha + beta)
        scale = alpha + beta

    so:
        alpha = scale * m
        beta  = scale * (1-m)
    """

    def __init__(self, invlink=inv_probit, scale=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = Parameter(scale, transform=positive())
        self.invlink = invlink

    def _scalar_density(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        beta = self.scale - alpha
        return logdensities.beta(Y, alpha, beta)

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.0)


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None, **kwargs):
        """
        A likelihood for multi-way classification.  Currently the only valid
        choice of inverse-link function (invlink) is an instance of RobustMax.

        For most problems, the stochastic `Softmax` likelihood may be more
        appropriate (note that you then cannot use Scipy optimizer).
        """
        super().__init__(num_latent_functions=num_classes, num_data_dims=None, **kwargs)
        self.num_classes = num_classes

        if invlink is None:
            invlink = RobustMax(self.num_classes)

        if not isinstance(invlink, RobustMax):
            raise NotImplementedError

        self.invlink = invlink

    def _log_prob(self, F, Y):
        hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
        yes = tf.ones(tf.shape(Y), dtype=default_float()) - self.invlink.epsilon
        no = tf.zeros(tf.shape(Y), dtype=default_float()) + self.invlink.eps_k1
        p = tf.where(hits, yes, no)
        return tf.reduce_sum(tf.math.log(p), axis=-1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        ve = p * tf.math.log(1.0 - self.invlink.epsilon) + (1.0 - p) * tf.math.log(
            self.invlink.eps_k1
        )
        return tf.reduce_sum(ve, axis=-1)

    def _predict_mean_and_var(self, Fmu, Fvar):
        possible_outputs = [
            tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64))
            for i in range(self.num_classes)
        ]
        ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
        return ps, ps - tf.square(ps)

    def _predict_density(self, Fmu, Fvar, Y):
        return tf.reduce_sum(tf.math.log(self._predict_non_logged_density(Fmu, Fvar, Y)), axis=-1)

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        den = p * (1.0 - self.invlink.epsilon) + (1.0 - p) * (self.invlink.eps_k1)
        return den

    def _conditional_mean(self, F):
        return self.invlink(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


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

    def check_last_dims_match(self, F, Y):
        tf.assert_equal(tf.shape(F)[-1], tf.shape(Y)[-1] - 1)

    def _scalar_density(self, F, Y):
        return self._partition_and_stitch([F, Y], "_scalar_density")

    def _predict_density(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "predict_density")

    def _variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], "variational_expectations")

    def _predict_mean_and_var(self, Fmu, Fvar):
        mvs = [lik.predict_mean_and_var(Fmu, Fvar) for lik in self.likelihoods]
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, 1)
        var = tf.concat(var_list, 1)
        return mu, var


class Ordinal(ScalarLikelihood):
    """
    A likelihood for doing ordinal regression.

    The data are integer values from 0 to K, and the user must specify (K-1)
    'bin edges' which define the points at which the labels switch. Let the bin
    edges be [a_0, a_1, ... a_{K-1}], then the likelihood is

    p(Y=0|F) = phi((a_0 - F) / sigma)
    p(Y=1|F) = phi((a_1 - F) / sigma) - phi((a_0 - F) / sigma)
    p(Y=2|F) = phi((a_2 - F) / sigma) - phi((a_1 - F) / sigma)
    ...
    p(Y=K|F) = 1 - phi((a_{K-1} - F) / sigma)

    where phi is the cumulative density function of a Gaussian (the inverse probit
    function) and sigma is a parameter to be learned. A reference is:

    @article{chu2005gaussian,
      title={Gaussian processes for ordinal regression},
      author={Chu, Wei and Ghahramani, Zoubin},
      journal={Journal of Machine Learning Research},
      volume={6},
      number={Jul},
      pages={1019--1041},
      year={2005}
    }
    """

    def __init__(self, bin_edges, **kwargs):
        """
        bin_edges is a numpy array specifying at which function value the
        output label should switch. If the possible Y values are 0...K, then
        the size of bin_edges should be (K-1).
        """
        super().__init__(**kwargs)
        self.bin_edges = bin_edges
        self.num_bins = bin_edges.size + 1
        self.sigma = Parameter(1.0, transform=positive())

    def _scalar_density(self, F, Y):
        Y = tf.cast(Y, default_int())
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        return tf.math.log(
            inv_probit(selected_bins_left - F / self.sigma)
            - inv_probit(selected_bins_right - F / self.sigma)
            + 1e-6
        )

    def _make_phi(self, F):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.

        Note that a matrix of F values is flattened.
        """
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        return inv_probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma) - inv_probit(
            scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma
        )

    def _conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()), (-1, 1))
        return tf.reshape(tf.linalg.matmul(phi, Ys), tf.shape(F))

    def _conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()), (-1, 1))
        E_y = phi @ Ys
        E_y2 = phi @ (Ys ** 2)
        return tf.reshape(E_y2 - E_y ** 2, tf.shape(F))


class MonteCarloLikelihood(Likelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_monte_carlo_points = 100
        del self.num_gauss_hermite_points

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

           \int\int y p(y|f)q(f) df dy

        and the predictive variance

           \int\int y^2 p(y|f)q(f) df dy  - [ \int\int y^2 p(y|f)q(f) df dy ]^2

        Here, we implement a default Monte Carlo routine.
        """
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(self.conditional_mean(*X))
        E_y, E_y2 = self._mc_quadrature(
            [self.conditional_mean, integrand2], Fmu, Fvar, epsilon=epsilon
        )
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y  # [N, D]

    def _predict_density(self, Fmu, Fvar, Y, epsilon=None):
        r"""
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df

        Here, we implement a default Monte Carlo routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, logspace=True, epsilon=epsilon), -1
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

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Monte Carlo quadrature routine.
        """
        return tf.reduce_sum(
            self._mc_quadrature(self.log_prob, Fmu, Fvar, Y=Y, epsilon=epsilon), axis=-1
        )


class GaussianMC(MonteCarloLikelihood, Gaussian):
    """
    Stochastic version of Gaussian likelihood for comparison.
    """

    pass


class Softmax(MonteCarloLikelihood):
    """
    The soft-max multi-class likelihood.  It can only provide a stochastic
    Monte-Carlo estimate of the variational expectations term, but this
    added variance tends to be small compared to that due to mini-batching
    (when using the SVGP model).
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(num_latent_functions=num_classes, num_data_dims=None, **kwargs)
        self.num_classes = self.num_latent_functions

    def _log_prob(self, F, Y):
        return -tf.nn.sparse_softmax_cross_entropy_with_logits(logits=F, labels=Y[:, 0])

    def _conditional_mean(self, F):
        return tf.nn.softmax(F)

    def _conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - p ** 2
