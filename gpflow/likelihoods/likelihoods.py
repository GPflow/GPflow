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
from ..base import Parameter, positive
from ..quadrature import hermgauss, ndiag_mc, ndiagquad
from ..util import default_float, default_int
from .robustmax import RobustMax


def inv_probit(x):
    jitter = 1e-3  # ensures output is strictly between 0 and 1
    return 0.5 * (1.0 + tf.math.erf(x / np.sqrt(2.0))) * (1 -
                                                          2 * jitter) + jitter


class Likelihood(tf.Module):
    def __init__(self):
        super().__init__()
        self.num_gauss_hermite_points = 20

    def predict_mean_and_var(self, Fmu, Fvar):
        """
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
            return self.conditional_variance(*X) + self.conditional_mean(*X)**2

        integrands = [self.conditional_mean, integrand]
        nghp = self.num_gauss_hermite_points
        E_y, E_y2 = ndiagquad(integrands, nghp, Fmu, Fvar)
        V_y = E_y2 - E_y**2
        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the log predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

            \log \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        integrand = self.log_prob
        nghp = self.num_gauss_hermite_points
        return ndiagquad(integrand, nghp, Fmu, Fvar, logspace=True, Y=Y)

    def variational_expectations(self, Fmu, Fvar, Y):
        """
        Compute the expected log density of the data, given a Gaussian
        distribution for the function values.

        if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes

           \int (\log p(y|f)) q(f) df.


        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        integrand = self.log_prob
        nghp = self.num_gauss_hermite_points
        return ndiagquad(integrand, nghp, Fmu, Fvar, Y=Y)


class Gaussian(Likelihood):
    def __init__(self, variance=1.0, **kwargs):
        super().__init__(**kwargs)
        self.variance = Parameter(variance, transform=positive())

    def log_prob(self, F, Y):
        return logdensities.gaussian(Y, F, self.variance)

    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    def conditional_variance(self, F):
        return tf.fill(F.shape, tf.squeeze(self.variance))

    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    def predict_density(self, Fmu, Fvar, Y):
        return logdensities.gaussian(Y, Fmu, Fvar + self.variance)

    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.math.log(self.variance) \
               - 0.5 * ((Y - Fmu) ** 2 + Fvar) / self.variance


class Poisson(Likelihood):
    """
    Poisson likelihood for use with count data, where the rate is given by the (transformed) GP.

    let g(.) be the inverse-link function, then this likelihood represents

    p(y_i | f_i) = Poisson(y_i | g(f_i) * binsize)

    Note:binsize
    For use in a Log Gaussian Cox process (doubly stochastic model) where the
    rate function of an inhomogeneous Poisson process is given by a GP.  The
    intractable likelihood can be approximated by gridding the space (into bins
    of size 'binsize') and using this Poisson likelihood.
    """

    def __init__(self, invlink=tf.exp, binsize=1., **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.binsize = np.array(binsize, dtype=default_float())

    def log_prob(self, F, Y):
        return logdensities.poisson(Y, self.invlink(F) * self.binsize)

    def conditional_variance(self, F):
        return self.invlink(F) * self.binsize

    def conditional_mean(self, F):
        return self.invlink(F) * self.binsize

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return Y * Fmu - tf.exp(Fmu + Fvar / 2) * self.binsize \
                   - tf.math.lgamma(Y + 1) + Y * tf.math.log(self.binsize)
        return super(Poisson, self).variational_expectations(Fmu, Fvar, Y)


class Exponential(Likelihood):
    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def log_prob(self, F, Y):
        return logdensities.exponential(Y, self.invlink(F))

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        return tf.square(self.invlink(F))

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return -tf.exp(-Fmu + Fvar / 2) * Y - Fmu
        return super().variational_expectations(Fmu, Fvar, Y)


class StudentT(Likelihood):
    def __init__(self, scale=1.0, df=3.0, **kwargs):
        """
        :param scale float: scale parameter
        :param df float: degrees of freedom
        """
        super().__init__(**kwargs)
        self.df = df
        self.scale = Parameter(scale,
                               transform=positive(),
                               dtype=default_float())

    def log_prob(self, F, Y):
        return logdensities.student_t(Y, F, self.scale, self.df)

    def conditional_mean(self, F):
        return F

    def conditional_variance(self, F):
        var = (self.scale**2) * (self.df / (self.df - 2.0))
        return tf.fill(F.shape, tf.squeeze(var))


class Bernoulli(Likelihood):
    def __init__(self, invlink=inv_probit, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink

    def log_prob(self, F, Y):
        return logdensities.bernoulli(Y, self.invlink(F))

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is inv_probit:
            p = inv_probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return super().predict_mean_and_var(Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return logdensities.bernoulli(Y, p)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - (p**2)


class Gamma(Likelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """

    def __init__(self, invlink=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink = invlink
        self.shape = Parameter(1.0, transform=positive())

    def log_prob(self, F, Y):
        return logdensities.gamma(Y, self.shape, self.invlink(F))

    def conditional_mean(self, F):
        return self.shape * self.invlink(F)

    def conditional_variance(self, F):
        scale = self.invlink(F)
        return self.shape * (scale**2)

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return -self.shape * Fmu - tf.math.lgamma(
                self.shape) + (self.shape - 1.) * tf.math.log(Y) - Y * tf.exp(
                    -Fmu + Fvar / 2.)
        else:
            return super().variational_expectations(Fmu, Fvar, Y)


class Beta(Likelihood):
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

    def log_prob(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        beta = self.scale - alpha
        return logdensities.beta(Y, alpha, beta)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.)


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None, **kwargs):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes

        if invlink is None:
            invlink = RobustMax(self.num_classes)

        if not isinstance(invlink, RobustMax):
            raise NotImplementedError

        self.invlink = invlink

    def log_prob(self, F, Y):
        hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1),
                        tf.cast(Y, tf.int64))
        yes = tf.ones(Y.shape, dtype=default_float()) - self.invlink.epsilon
        no = tf.zeros(Y.shape, dtype=default_float()) + self.invlink.eps_k1
        p = tf.where(hits, yes, no)
        return tf.math.log(p)

    def variational_expectations(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        ve = p * tf.math.log(1. - self.invlink.epsilon) + (
            1. - p) * tf.math.log(self.invlink.eps_k1)
        return ve

    def predict_mean_and_var(self, Fmu, Fvar):
        possible_outputs = [
            tf.fill(tf.stack([Fmu.shape[0], 1]), np.array(i, dtype=np.int64))
            for i in range(self.num_classes)
        ]
        ps = [
            self._predict_non_logged_density(Fmu, Fvar, po)
            for po in possible_outputs
        ]
        ps = tf.transpose(tf.stack([tf.reshape(p, (-1, )) for p in ps]))
        return ps, ps - tf.square(ps)

    def predict_density(self, Fmu, Fvar, Y):
        return tf.math.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
        den = p * (1. - self.invlink.epsilon) + (1. -
                                                 p) * (self.invlink.eps_k1)
        return den

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


class SwitchedLikelihood(Likelihood):
    def __init__(self, likelihood_list, **kwargs):
        """
        In this likelihood, we assume at extra column of Y, which contains
        integers that specify a likelihood from the list of likelihoods.
        """
        super().__init__(**kwargs)
        for l in likelihood_list:
            assert isinstance(l, Likelihood)
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
        ind = Y[:, -1]
        ind = tf.cast(ind, tf.int32)
        Y = Y[:, :-1]
        args[-1] = Y

        # split up the arguments into chunks corresponding to the relevant likelihoods
        args = zip(*[
            tf.dynamic_partition(X, ind, len(self.likelihoods)) for X in args
        ])

        # apply the likelihood-function to each section of the data
        funcs = [getattr(lik, func_name) for lik in self.likelihoods]
        results = [f(*args_i) for f, args_i in zip(funcs, args)]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind,
                                          len(self.likelihoods))
        results = tf.dynamic_stitch(partitions, results)

        return results

    def log_prob(self, F, Y):
        return self._partition_and_stitch([F, Y], 'log_prob')

    def predict_density(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], 'predict_density')

    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y],
                                          'variational_expectations')

    def predict_mean_and_var(self, Fmu, Fvar):
        mvs = [lik.predict_mean_and_var(Fmu, Fvar) for lik in self.likelihoods]
        mu_list, var_list = zip(*mvs)
        mu = tf.concat(mu_list, 1)
        var = tf.concat(var_list, 1)
        return mu, var


class Ordinal(Likelihood):
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

    def log_prob(self, F, Y):
        Y = tf.cast(Y, default_int())
        scaled_bins_left = tf.concat(
            [self.bin_edges / self.sigma,
             np.array([np.inf])], 0)
        scaled_bins_right = tf.concat(
            [np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        return tf.math.log(
            inv_probit(selected_bins_left - F / self.sigma) -
            inv_probit(selected_bins_right - F / self.sigma) + 1e-6)

    def _make_phi(self, F):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.

        Note that a matrix of F values is flattened.
        """
        scaled_bins_left = tf.concat(
            [self.bin_edges / self.sigma,
             np.array([np.inf])], 0)
        scaled_bins_right = tf.concat(
            [np.array([-np.inf]), self.bin_edges / self.sigma], 0)
        return inv_probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma) \
               - inv_probit(scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma)

    def conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()),
                        (-1, 1))
        return tf.reshape(tf.linalg.matmul(phi, Ys), F.shape)

    def conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=default_float()),
                        (-1, 1))
        E_y = phi @ Ys
        E_y2 = phi @ (Ys**2)
        return tf.reshape(E_y2 - E_y**2, F.shape)


class MonteCarloLikelihood(Likelihood):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_monte_carlo_points = 100
        del self.num_gauss_hermite_points

    def _mc_quadrature(self,
                       funcs,
                       Fmu,
                       Fvar,
                       logspace: bool = False,
                       epsilon=None,
                       **Ys):
        return ndiag_mc(funcs, self.num_monte_carlo_points, Fmu, Fvar,
                        logspace, epsilon, **Ys)

    def predict_mean_and_var(self, Fmu, Fvar, epsilon=None):
        """
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
        integrand2 = lambda *X: self.conditional_variance(*X) + tf.square(
            self.conditional_mean(*X))
        E_y, E_y2 = self._mc_quadrature([self.conditional_mean, integrand2],
                                        Fmu,
                                        Fvar,
                                        epsilon=epsilon)
        V_y = E_y2 - tf.square(E_y)
        return E_y, V_y  # [N, D]

    def predict_density(self, Fmu, Fvar, Y, epsilon=None):
        """
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
        return self._mc_quadrature(self.log_prob,
                                   Fmu,
                                   Fvar,
                                   Y=Y,
                                   logspace=True,
                                   epsilon=epsilon)

    def variational_expectations(self, Fmu, Fvar, Y, epsilon=None):
        """
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
        return self._mc_quadrature(self.log_prob,
                                   Fmu,
                                   Fvar,
                                   Y=Y,
                                   epsilon=epsilon)


class GaussianMC(MonteCarloLikelihood, Gaussian):
    """
    Stochastic version of Gaussian likelihood for comparison.
    """
    pass


class Softmax(MonteCarloLikelihood):
    """
    The soft-max multi-class likelihood.
    """

    def __init__(self, num_classes, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes

    def log_prob(self, F, Y):
        with tf.control_dependencies([
                tf.assert_equal(Y.shape[1], 1),
                tf.assert_equal(F.shape[1], self.num_classes)
        ]):
            return -tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=F, labels=Y[:, 0])[:, None]

    def conditional_mean(self, F):
        return tf.nn.softmax(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - p**2
