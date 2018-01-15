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


from __future__ import absolute_import

import tensorflow as tf
import numpy as np

from . import settings
from . import densities
from . import transforms

from .decors import params_as_tensors
from .decors import params_as_tensors_for
from .params import Parameter
from .params import Parameterized
from .params import ParamList
from .quadrature import hermgauss


class Likelihood(Parameterized):
    def __init__(self, name=None):
        super(Likelihood, self).__init__(name)
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
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_w /= np.sqrt(np.pi)
        gh_w = gh_w.reshape(-1, 1)
        shape = tf.shape(Fmu)
        Fmu, Fvar = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        # here's the quadrature for the mean
        E_y = tf.reshape(tf.matmul(self.conditional_mean(X), gh_w), shape)

        # here's the quadrature for the variance
        integrand = self.conditional_variance(X) \
            + tf.square(self.conditional_mean(X))
        V_y = tf.reshape(tf.matmul(integrand, gh_w), shape) - tf.square(E_y)

        return E_y, V_y

    def predict_density(self, Fmu, Fvar, Y):
        """
        Given a Normal distribution for the latent function, and a datum Y,
        compute the (log) predictive density of Y.

        i.e. if
            q(f) = N(Fmu, Fvar)

        and this object represents

            p(y|f)

        then this method computes the predictive density

           \int p(y=Y|f)q(f) df

        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        """
        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)

        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x[None, :] * tf.sqrt(2.0 * Fvar) + Fmu

        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.log(tf.matmul(tf.exp(logp), gh_w)), shape)

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

        gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
        gh_x = gh_x.reshape(1, -1)
        gh_w = gh_w.reshape(-1, 1) / np.sqrt(np.pi)
        shape = tf.shape(Fmu)
        Fmu, Fvar, Y = [tf.reshape(e, (-1, 1)) for e in (Fmu, Fvar, Y)]
        X = gh_x * tf.sqrt(2.0 * Fvar) + Fmu
        Y = tf.tile(Y, [1, self.num_gauss_hermite_points])  # broadcast Y to match X

        logp = self.logp(X, Y)
        return tf.reshape(tf.matmul(logp, gh_w), shape)


class Gaussian(Likelihood):
    def __init__(self, var=1.0):
        super().__init__()
        self.variance = Parameter(
            var, transform=transforms.positive, dtype=settings.float_type)

    @params_as_tensors
    def logp(self, F, Y):
        return densities.gaussian(F, Y, self.variance)

    @params_as_tensors
    def conditional_mean(self, F):  # pylint: disable=R0201
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F):
        return tf.fill(tf.shape(F), tf.squeeze(self.variance))

    @params_as_tensors
    def predict_mean_and_var(self, Fmu, Fvar):
        return tf.identity(Fmu), Fvar + self.variance

    @params_as_tensors
    def predict_density(self, Fmu, Fvar, Y):
        return densities.gaussian(Fmu, Y, Fvar + self.variance)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        return -0.5 * np.log(2 * np.pi) - 0.5 * tf.log(self.variance) \
               - 0.5 * (tf.square(Y - Fmu) + Fvar) / self.variance


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

    def __init__(self, invlink=tf.exp, binsize=1.):
        Likelihood.__init__(self)
        self.invlink = invlink
        self.binsize = np.double(binsize)

    def logp(self, F, Y):
        return densities.poisson(self.invlink(F) * self.binsize, Y)

    def conditional_variance(self, F):
        return self.invlink(F) * self.binsize

    def conditional_mean(self, F):
        return self.invlink(F) * self.binsize

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return Y * Fmu - tf.exp(Fmu + Fvar / 2) * self.binsize \
                   - tf.lgamma(Y + 1) + Y * tf.log(self.binsize)
        return super(Poisson, self).variational_expectations(Fmu, Fvar, Y)

class Exponential(Likelihood):
    def __init__(self, invlink=tf.exp):
        super().__init__()
        self.invlink = invlink

    def logp(self, F, Y):
        return densities.exponential(self.invlink(F), Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        return tf.square(self.invlink(F))

    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return - tf.exp(-Fmu + Fvar / 2) * Y - Fmu
        return super().variational_expectations(Fmu, Fvar, Y)


class StudentT(Likelihood):
    def __init__(self, deg_free=3.0):
        Likelihood.__init__(self)
        self.deg_free = deg_free
        self.scale = Parameter(1.0, transform=transforms.positive)

    @params_as_tensors
    def logp(self, F, Y):
        return densities.student_t(Y, F, self.scale, self.deg_free)

    @params_as_tensors
    def conditional_mean(self, F):
        return tf.identity(F)

    @params_as_tensors
    def conditional_variance(self, F):
        return F * 0.0 + (self.deg_free / (self.deg_free - 2.0))


def probit(x):
    return 0.5 * (1.0 + tf.erf(x / np.sqrt(2.0))) * (1 - 2e-3) + 1e-3


class Bernoulli(Likelihood):
    def __init__(self, invlink=probit):
        Likelihood.__init__(self)
        self.invlink = invlink

    def logp(self, F, Y):
        return densities.bernoulli(self.invlink(F), Y)

    def predict_mean_and_var(self, Fmu, Fvar):
        if self.invlink is probit:
            p = probit(Fmu / tf.sqrt(1 + Fvar))
            return p, p - tf.square(p)
        else:
            # for other invlink, use quadrature
            return Likelihood.predict_mean_and_var(self, Fmu, Fvar)

    def predict_density(self, Fmu, Fvar, Y):
        p = self.predict_mean_and_var(Fmu, Fvar)[0]
        return densities.bernoulli(p, Y)

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.invlink(F)
        return p - tf.square(p)


class Gamma(Likelihood):
    """
    Use the transformed GP to give the *scale* (inverse rate) of the Gamma
    """

    def __init__(self, invlink=tf.exp):
        Likelihood.__init__(self)
        self.invlink = invlink
        self.shape = Parameter(1.0, transform=transforms.positive)

    @params_as_tensors
    def logp(self, F, Y):
        return densities.gamma(self.shape, self.invlink(F), Y)

    @params_as_tensors
    def conditional_mean(self, F):
        return self.shape * self.invlink(F)

    @params_as_tensors
    def conditional_variance(self, F):
        scale = self.invlink(F)
        return self.shape * tf.square(scale)

    @params_as_tensors
    def variational_expectations(self, Fmu, Fvar, Y):
        if self.invlink is tf.exp:
            return -self.shape * Fmu - tf.lgamma(self.shape) \
                + (self.shape - 1.) * tf.log(Y) - Y * tf.exp(-Fmu + Fvar / 2.)
        else:
            return Likelihood.variational_expectations(self, Fmu, Fvar, Y)


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

    def __init__(self, invlink=probit, scale=1.0):
        Likelihood.__init__(self)
        self.scale = Parameter(scale, transform=transforms.positive)
        self.invlink = invlink

    @params_as_tensors
    def logp(self, F, Y):
        mean = self.invlink(F)
        alpha = mean * self.scale
        beta = self.scale - alpha
        return densities.beta(alpha, beta, Y)

    @params_as_tensors
    def conditional_mean(self, F):
        return self.invlink(F)

    @params_as_tensors
    def conditional_variance(self, F):
        mean = self.invlink(F)
        return (mean - tf.square(mean)) / (self.scale + 1.)


class RobustMax(object):
    """
    This class represent a multi-class inverse-link function. Given a vector
    f=[f_1, f_2, ... f_k], the result of the mapping is

    y = [y_1 ... y_k]

    with

    y_i = (1-eps)  i == argmax(f)
          eps/(k-1)  otherwise.


    """

    def __init__(self, num_classes, epsilon=1e-3):
        self.epsilon = epsilon
        self.num_classes = num_classes
        self._eps_K1 = self.epsilon / (self.num_classes - 1.)

    def __call__(self, F):
        i = tf.argmax(F, 1)
        return tf.one_hot(i, self.num_classes, 1. - self.epsilon, self._eps_K1)

    def prob_is_largest(self, Y, mu, var, gh_x, gh_w):
        Y = tf.cast(Y, tf.int64)
        # work out what the mean and variance is of the indicated latent function.
        oh_on = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 1., 0.), settings.float_type)
        mu_selected = tf.reduce_sum(oh_on * mu, 1)
        var_selected = tf.reduce_sum(oh_on * var, 1)

        # generate Gauss Hermite grid
        X = tf.reshape(mu_selected, (-1, 1)) + gh_x * tf.reshape(
            tf.sqrt(tf.clip_by_value(2. * var_selected, 1e-10, np.inf)), (-1, 1))

        # compute the CDF of the Gaussian between the latent functions and the grid (including the selected function)
        dist = (tf.expand_dims(X, 1) - tf.expand_dims(mu, 2)) / tf.expand_dims(
            tf.sqrt(tf.clip_by_value(var, 1e-10, np.inf)), 2)
        cdfs = 0.5 * (1.0 + tf.erf(dist / np.sqrt(2.0)))

        cdfs = cdfs * (1 - 2e-4) + 1e-4

        # blank out all the distances on the selected latent function
        oh_off = tf.cast(tf.one_hot(tf.reshape(Y, (-1,)), self.num_classes, 0., 1.), settings.float_type)
        cdfs = cdfs * tf.expand_dims(oh_off, 2) + tf.expand_dims(oh_on, 2)

        # take the product over the latent functions, and the sum over the GH grid.
        return tf.matmul(tf.reduce_prod(cdfs, reduction_indices=[1]), tf.reshape(gh_w / np.sqrt(np.pi), (-1, 1)))


class MultiClass(Likelihood):
    def __init__(self, num_classes, invlink=None):
        """
        A likelihood that can do multi-way classification.
        Currently the only valid choice
        of inverse-link function (invlink) is an instance of RobustMax.
        """
        Likelihood.__init__(self)
        self.num_classes = num_classes
        if invlink is None:
            invlink = RobustMax(self.num_classes)
        elif not isinstance(invlink, RobustMax):
            raise NotImplementedError
        self.invlink = invlink

    def logp(self, F, Y):
        if isinstance(self.invlink, RobustMax):
            hits = tf.equal(tf.expand_dims(tf.argmax(F, 1), 1), tf.cast(Y, tf.int64))
            yes = tf.ones(tf.shape(Y), dtype=settings.float_type) - self.invlink.epsilon
            no = tf.zeros(tf.shape(Y), dtype=settings.float_type) + self.invlink._eps_K1
            p = tf.where(hits, yes, no)
            return tf.log(p)
        else:
            raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * np.log(1 - self.invlink.epsilon) + (1. - p) * np.log(self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def predict_mean_and_var(self, Fmu, Fvar):
        if isinstance(self.invlink, RobustMax):
            # To compute this, we'll compute the density for each possible output
            possible_outputs = [tf.fill(tf.stack([tf.shape(Fmu)[0], 1]), np.array(i, dtype=np.int64)) for i in
                                range(self.num_classes)]
            ps = [self._predict_non_logged_density(Fmu, Fvar, po) for po in possible_outputs]
            ps = tf.transpose(tf.stack([tf.reshape(p, (-1,)) for p in ps]))
            return ps, ps - tf.square(ps)
        else:
            raise NotImplementedError

    def predict_density(self, Fmu, Fvar, Y):
        return tf.log(self._predict_non_logged_density(Fmu, Fvar, Y))

    def _predict_non_logged_density(self, Fmu, Fvar, Y):
        if isinstance(self.invlink, RobustMax):
            gh_x, gh_w = hermgauss(self.num_gauss_hermite_points)
            p = self.invlink.prob_is_largest(Y, Fmu, Fvar, gh_x, gh_w)
            return p * (1 - self.invlink.epsilon) + (1. - p) * (self.invlink._eps_K1)
        else:
            raise NotImplementedError

    def conditional_mean(self, F):
        return self.invlink(F)

    def conditional_variance(self, F):
        p = self.conditional_mean(F)
        return p - tf.square(p)


class SwitchedLikelihood(Likelihood):
    def __init__(self, likelihood_list):
        """
        In this likelihood, we assume at extra column of Y, which contains
        integers that specify a likelihood from the list of likelihoods.
        """
        Likelihood.__init__(self)
        for l in likelihood_list:
            assert isinstance(l, Likelihood)
        self.likelihood_list = ParamList(likelihood_list)
        self.num_likelihoods = len(self.likelihood_list)

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
        args = zip(*[tf.dynamic_partition(X, ind, self.num_likelihoods) for X in args])

        # apply the likelihood-function to each section of the data
        with params_as_tensors_for(self, convert=False):
            funcs = [getattr(lik, func_name) for lik in self.likelihood_list]
        results = [f(*args_i) for f, args_i in zip(funcs, args)]

        # stitch the results back together
        partitions = tf.dynamic_partition(tf.range(0, tf.size(ind)), ind, self.num_likelihoods)
        results = tf.dynamic_stitch(partitions, results)

        return results

    def logp(self, F, Y):
        return self._partition_and_stitch([F, Y], 'logp')

    def predict_density(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], 'predict_density')

    def variational_expectations(self, Fmu, Fvar, Y):
        return self._partition_and_stitch([Fmu, Fvar, Y], 'variational_expectations')

    def predict_mean_and_var(self, Fmu, Fvar):
        mvs = [lik.predict_mean_and_var(Fmu, Fvar) for lik in self.likelihood_list]
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

    where phi is the cumulative density function of a Gaussian (the probit
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
    def __init__(self, bin_edges):
        """
        bin_edges is a numpy array specifying at which function value the
        output label should switch. In the possible Y values are 0...K, then
        the size of bin_edges should be (K-1).
        """
        Likelihood.__init__(self)
        self.bin_edges = bin_edges
        self.num_bins = bin_edges.size + 1
        self.sigma = Parameter(1.0, transform=transforms.positive)

    @params_as_tensors
    def logp(self, F, Y):
        Y = tf.cast(Y, tf.int64)
        scaled_bins_left = tf.concat([self.bin_edges/self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges/self.sigma], 0)
        selected_bins_left = tf.gather(scaled_bins_left, Y)
        selected_bins_right = tf.gather(scaled_bins_right, Y)

        return tf.log(probit(selected_bins_left - F / self.sigma) -
                      probit(selected_bins_right - F / self.sigma) + 1e-6)

    @params_as_tensors
    def _make_phi(self, F):
        """
        A helper function for making predictions. Constructs a probability
        matrix where each row output the probability of the corresponding
        label, and the rows match the entries of F.

        Note that a matrix of F values is flattened.
        """
        scaled_bins_left = tf.concat([self.bin_edges / self.sigma, np.array([np.inf])], 0)
        scaled_bins_right = tf.concat([np.array([-np.inf]), self.bin_edges/self.sigma], 0)
        return probit(scaled_bins_left - tf.reshape(F, (-1, 1)) / self.sigma)\
            - probit(scaled_bins_right - tf.reshape(F, (-1, 1)) / self.sigma)

    def conditional_mean(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        return tf.reshape(tf.matmul(phi, Ys), tf.shape(F))

    def conditional_variance(self, F):
        phi = self._make_phi(F)
        Ys = tf.reshape(np.arange(self.num_bins, dtype=np.float64), (-1, 1))
        E_y = tf.matmul(phi, Ys)
        E_y2 = tf.matmul(phi, tf.square(Ys))
        return tf.reshape(E_y2 - tf.square(E_y), tf.shape(F))
