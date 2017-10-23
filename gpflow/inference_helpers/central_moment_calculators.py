# Copyright 2017 John Bradshaw
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

import tensorflow as tf
import numpy as np

from .. import gaussian_utils
from .. import settings


class BaseCentralMomentCalculator(object):
    """
    Base class defining interface of calculating moments of functions with one dimensional
    Gaussians. Note parametrise the Gaussian with their natural parameters.
    """
    def calc_first_and_second_centmoments(self, tau_mi, nu_mi):
        """
        Calculate the 1st and 2nd moments between the Gaussian defined by the natural parameters
        tau_mi and nu_mi and the function of this class.
        works element-wise
        """
        first_moment, second_moment = self._calc_first_and_second_centmoments(tau_mi, nu_mi)
        return tf.identity(first_moment, name="first_moment"), \
               tf.identity(second_moment, name="second_moment")

    def calc_log_zero_moment(self, tau_mi, nu_mi):
        """
        Calculates the zeroth moment between the Gaussian parameterised by tau and nu and the
        function represented by the class.
        work elementwise
        """
        raise NotImplementedError

    def _calc_first_and_second_centmoments(self, tau_mi, nu_mi):
        raise NotImplementedError


class StepFunction(BaseCentralMomentCalculator):
    """
    Calcs the moments between Gaussians and heavyside step functions.
    By default we assume that the heavy side step function starts high and goes to zero at
    `threshold` ie we have one side truncatiuon of the Gaussian at the upper tail.
    However, the opposite behaviour can be found by start_high being set to False.
    Note that the resulting distribution will be a Truncated Normal for which the moments
    are defined at https://en.wikipedia.org/wiki/Truncated_normal_distribution
    """
    def __init__(self, threshold, start_high=True):
        self.threshold = threshold
        self.start_high = start_high

    def _calc_first_and_second_centmoments(self, tau_mi, nu_mi):
        beta, sigma_sq_mi, mu_mi = self._calculate_beta_and_sigma_sq_i_and_mu_i(tau_mi, nu_mi)
        neg_factor = float(self.start_high == True) * 2 - 1.
        phi_beta_over_cdf_term = gaussian_utils.deriv_log_cdf_normal(neg_factor *beta)
        first_moment = mu_mi - neg_factor * tf.sqrt(sigma_sq_mi) * phi_beta_over_cdf_term
        second_moment = sigma_sq_mi * (
            1 - neg_factor * beta * phi_beta_over_cdf_term - tf.square(phi_beta_over_cdf_term))
        # NB: Wikipedia says there may be a simpler way to get second moment from
        # Barr and Sherrill (1999)
        return first_moment, second_moment

    def calc_log_zero_moment(self, tau_mi, nu_mi):
        gaussian_dist = tf.distributions.Normal(settings.np_float(0.), settings.np_float(1.))
        beta, _, _ = self._calculate_beta_and_sigma_sq_i_and_mu_i(tau_mi, nu_mi)
        neg_factor = float(self.start_high == True) * 2 - 1.
        log_cdf = gaussian_dist.log_cdf(neg_factor * beta)
        return log_cdf

    def _calculate_beta_and_sigma_sq_i_and_mu_i(self, tau_mi, nu_mi):
        sigma_sq_mi = tf.reciprocal(tau_mi)
        mu_mi = nu_mi * sigma_sq_mi
        beta = (self.threshold - mu_mi) * tf.sqrt(tau_mi)
        return beta, sigma_sq_mi, mu_mi


class UnormalisedGaussian(BaseCentralMomentCalculator):
    """
    Moments with respect to
    .. math::
      Z \\mathcal N (\\mu \\sigma^2)

    Note that when you multiply two Gaussians together you get a new un-normalised
    Gaussian with summed natural parameters.

    This moment function is of use when checking EP code runs correctly.
    """
    def __init__(self, factor, mu, sigma):
        """
        Parameters can either be scalars or Tensors the same size as the data that will be fed in.
        """
        self.factor = factor
        self.mu = mu
        self.sigma = sigma

    def _calc_first_and_second_centmoments(self, tau_mi, nu_mi):
        nu_0, tau_0 = gaussian_utils.convert_from_params_to_natural(self.mu, tf.square(self.sigma))

        tau_final = tau_0 + tau_mi
        nu_final = nu_0 + nu_mi
        mu_final, sigma_sq_final = gaussian_utils.convert_from_natural_to_params(nu_final, tau_final)

        first_moment = mu_final

        second_moment = sigma_sq_final
        return first_moment, second_moment

    def calc_log_zero_moment(self, tau_mi, nu_mi):
        mu_mi, sigma_sq_mi = gaussian_utils.convert_from_natural_to_params(nu_mi, tau_mi)
        log_zep = self._log_z_helpers(sigma_sq_mi, mu_mi)
        return log_zep

    def _log_z_helpers(self, sigma_sq_mi, mu_mi):
        log_factor = tf.log(self.factor)
        log_term_one = -0.5 * np.log(2. * np.pi, dtype=settings.np_float)
        added_sigma_sq = sigma_sq_mi + self.sigma**2
        log_term_two = -0.5 * tf.log(added_sigma_sq)
        log_term_three = -0.5 * tf.square(self.mu - mu_mi) / added_sigma_sq
        return log_factor + (log_term_one + log_term_two + log_term_three)

    def _z_helpers(self, sigma_sq_mi, mu_mi):
        factor_one = 1./np.sqrt(2.*np.pi)
        added_sigmas_sq = sigma_sq_mi + tf.square(self.sigma)
        factor_two = tf.reciprocal(tf.sqrt(added_sigmas_sq))
        factor_three = tf.exp(-0.5 * tf.square(self.mu - mu_mi) / added_sigmas_sq)
        return self.factor *(factor_one * factor_two * factor_three)


class CDFNormalGaussian(BaseCentralMomentCalculator):
    """
    Central with respect to the cdf of  Gaussian, ie with respect to

    .. math::
      \\phi \\left( \\frac{y-m}{v} \\right)

    where

    .. math::
      \\phi(x) = \int^{x}_{-\\infty} \\mathcal N(t| 0, 1) dt
    The moments are defined in Section 3.9 of
    ::
      @book{rasmussen2006gaussian,
        title={Gaussian processes for machine learning},
        author={Rasmussen, Carl Edward and Williams, Christopher KI},
        volume={1},
        year={2006},
        publisher={MIT press Cambridge}
      }

    We allow v to be negative.
    """
    def __init__(self, m, v):
        self.m = m
        self.v = v  # nb note v corresponds to a std dev

    def _calc_first_and_second_centmoments(self, tau_mi, nu_mi):
        mu_mi, sigma_sq_mi = gaussian_utils.convert_from_natural_to_params(nu_mi, tau_mi)
        z = self._calc_z(mu_mi, sigma_sq_mi)

        norm_over_cdf_z = gaussian_utils.deriv_log_cdf_normal(z)
        under_factor = tf.identity(self.v * tf.sqrt((1 + sigma_sq_mi/self.v**2)), name="under_factor")
        first_moment = mu_mi + sigma_sq_mi * norm_over_cdf_z / under_factor

        second_term_second_factor = z + norm_over_cdf_z
        second_moment = sigma_sq_mi - norm_over_cdf_z * sigma_sq_mi**2 / (self.v**2 + sigma_sq_mi) * second_term_second_factor
        return first_moment, second_moment

    def calc_log_zero_moment(self, tau_mi, nu_mi):
        mu_mi, sigma_sq_mi = gaussian_utils.convert_from_natural_to_params(nu_mi, tau_mi)
        z = self._calc_z(mu_mi, sigma_sq_mi)

        normal = tf.distributions.Normal(loc=settings.np_float(0.), scale=settings.np_float(1.))
        log_cdf_z = normal.log_cdf(z)
        return log_cdf_z

    def _calc_z(self, mu_mi, sigma_sq_mi):
        return tf.identity((mu_mi - self.m) / (self.v * tf.sqrt(1 + sigma_sq_mi/(self.v**2))),
                           name="z_calculation")
