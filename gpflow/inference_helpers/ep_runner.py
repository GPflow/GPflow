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

from __future__ import absolute_import

import collections

import tensorflow as tf

from gpflow.params import Parameter
from gpflow.params import Parameterized
from gpflow.decors import name_scope
from gpflow.decors import params_as_tensors
from gpflow import settings


VERY_VERY_SMALL_NUMBER = 1e-20


class EPRunner(Parameterized):
    """
    Expectation Propagation (EP) runner for evaluating the parameters of Gaussian factors
    used to approximate factors that lead to intractable expressions.

    An example would be GP binary classification Here we have
    .. math::
      p(f|X,y) = 1/Z p(f|X) \\prod_i p(y_i|f_i)

    This is intractable in classification where the likelihood is for instance the Std Normal CDF.
    So instead we approximate each one of these factors by an un-normalised Gaussian, these
    parameters are often denoted as t_i. The parameters for the Gaussians are found via EP.

    Note we often parameterise the Gaussians by their natural parameters:
    nu = 1/sigma^2 * mu
    tau = 1 / sigma^2

    NB currently runs parallel EP. Ie updates all points at once.

    The reference for EP is:
    ::
      @inproceedings{minka2001expectation,
          title={Expectation propagation for approximate Bayesian inference},
          author={Minka, Thomas P},
          booktitle={Proceedings of the Seventeenth conference on Uncertainty in
           artificial intelligence},
          pages={362--369},
          year={2001},
          organization={Morgan Kaufmann Publishers Inc.}
      }

    Its use in binary classification is well described by:
    ::
      @book{rasmussen2006gaussian,
        title={Gaussian processes for machine learning},
        author={Rasmussen, Carl Edward and Williams, Christopher KI},
        volume={1},
        year={2006},
        publisher={MIT press Cambridge}
      }

    Note this currently will probably only work on one dimensional latent functions.
    """

    EpParams = collections.namedtuple("EpParams", ["old_tau_tilde", "old_nu_tilde",
                                                   "new_tau_tilde", "new_nu_tilde",
                                                   "sigma", "mu", "counter"])

    def __init__(self, moment_function_calculator, max_iter=100, convergent_tol=1e-6):
        """
        :param central_moments_calculator: a function to create a
        class that contains a function that computes and returns
        the first and second central moment of this function against one-d Gaussians
        that are parametrised elementise by taus and nus
        that are fed into this function.
        calc_first_and_second_moments(tau_vector, nu_vector) => (first_moments, second_moments)
        and also calculates the  zeroth moment
        calc_log_zero_moment(tau_vector, nu_vector) => zeroth moment
        :param max_iter: the maximum number of EP iterations to try
        :param convergent_tol: quits EP iterations early when the avg squared difference of
         both the tau and nu changes falls below this value.
        """
        super().__init__()
        self.central_moments_calculator = moment_function_calculator
        self.convergent_tol = Parameter(settings.np_float(convergent_tol), trainable=False, name="convergent_tol")
        self.max_iter = Parameter(settings.np_int(max_iter), trainable=False, name="Max_ep_iters")

    @params_as_tensors
    def run_ep(self, f_mu_pre, f_var_pre, nu_tilde_pre, tau_tilde_pre):
        """
        We are going to approximate the following equation.
        .. math::
          \\mathcal p (f| \\ldots) \\approx 1/Z \\mathcal N (\\mu_0, \\Sigma_0) \\mathcal (\\tilde \\mu, \\tilde \\Sigma)
        This function runs EP and return more up to date tilde parameters as well as the final sigma
        and mu.
        Note that the the gradients are stopped at the end of this function. This stops them from
        having to be calculated back through the entire loop. So if you want to get gradients for
        the final Sigma wrt Sigma_0 then you will need to recompute it with the final tau_tilde
        and mu_tilde parameters.
        :param f_mu_pre: mu0, column vecor
        :param f_var_pre: Sigma0, 2d matrix
        :param nu_tilde_pre: initial guess for nu_tilde parmaters, column vector
        :param tau_tilde_pre: initial guess for tau tilde parameters, column vector
        """
        ep_params = self.EpParams(old_tau_tilde=tf.zeros_like(tau_tilde_pre, dtype=settings.tf_float),
                                  old_nu_tilde=tf.zeros_like(nu_tilde_pre, dtype=settings.tf_float),
                                  new_tau_tilde=tau_tilde_pre,
                                  new_nu_tilde=nu_tilde_pre,
                                  sigma=f_var_pre,
                                  mu=f_mu_pre,
                                  counter=tf.constant(0, settings.tf_int))

        # We now define the condition and body for our while loop
        def condition(ep_params):
            just_started_flag = tf.less(ep_params.counter, 1)
            nu_sq_chg = _create_mean_square_change(ep_params.old_nu_tilde,
                                                   ep_params.new_nu_tilde, name="sq_nu_change")
            tau_sq_chg = _create_mean_square_change(ep_params.old_tau_tilde,
                                                    ep_params.new_tau_tilde, name="sq_tau_change")
            converged_flag = tf.logical_and(tf.less_equal(nu_sq_chg, self.convergent_tol),
                                            tf.less_equal(tau_sq_chg, self.convergent_tol),
                                            "ep_converged")

            below_max_iter_and_not_converged = tf.logical_and(tf.less(ep_params.counter, self.max_iter),
                                                              tf.logical_not(converged_flag))
            continue_flag = tf.logical_or(just_started_flag,
                                          below_max_iter_and_not_converged
                                          , name="continue_flag")
            return continue_flag

        def body(ep_params):
            # We run parallel EP.
            tau_mi, nu_mi = self._create_cavity_dist(ep_params.sigma, ep_params.mu,
                                                ep_params.new_tau_tilde,
                                                ep_params.new_nu_tilde)

            mu_hat, sigma_sq_hat = self.central_moments_calculator().calc_first_and_second_centmoments(tau_mi, nu_mi)
            tau_tilde_newest, nu_tilde_newest = self._update_tilde_params(sigma_sq_hat, mu_hat, tau_mi,
                                                                     nu_mi)
            Sigma_new, mu_new = self.compute_newest_sigma_and_mu(tau_tilde_newest, nu_tilde_newest,
                                                                 f_var_pre, f_mu_pre)
            new_ep_params = self.EpParams(old_tau_tilde=ep_params.new_tau_tilde,
                                          old_nu_tilde=ep_params.new_nu_tilde,
                                          new_tau_tilde=tau_tilde_newest,
                                          new_nu_tilde=nu_tilde_newest,
                                          sigma=Sigma_new,
                                          mu=mu_new,
                                          counter=tf.add(ep_params.counter, 1, name="counter_add")
                                          )
            return (new_ep_params,)

        # And now we run the loop!
        final_params = tf.while_loop(cond=condition, body=body,
                                     loop_vars=(ep_params,), back_prop=False)[0]

        # We will stop the gradients here from flowing back through.
        return_results = self.EpParams(*(tf.stop_gradient(v) for v in final_params))
        return return_results

    def calculate_log_normalising_constant(self,  f_mu_pre, f_var_pre, nu_tilde, tau_tilde):
        """
        Here we are going to compute the normalising constant for EP.
        To summarise we have
        .. math::
          p (f| y, \\ldots) \\approx 1/Z_{EP} \\mathcal N (\\mu_0, \\Sigma_0) \prod_i t_i(\\tilde Z_i, \\tilde \\mu_i, \\tilde \\Sigma_i)
        And we are interested here in finding Z_{EP} as this is an approximation for
        $ p(y | \\ldots)$ which in many cases will be the marginal likelihood and so something we
        are interested in optimising. Note that we return the log of this.

        For details of how this can be computed in a stable way please see Section 3.6 of
        ::
          @book{rasmussen2006gaussian,
            title={Gaussian processes for machine learning},
            author={Rasmussen, Carl Edward and Williams, Christopher KI},
            volume={1},
            year={2006},
            publisher={MIT press Cambridge}
          }
        We will refer to the equations of GPML in this method. Mostly we follow Eqn 3.65 and so name
        our terms after those ones.

        f_mu_pre, nu_tilde, tau_tilde should be column vectors.
        f_var_pre should be a two dimensional Tensor.
        """
        # Latest Sigma
        Sigma_final, mu_final = self.compute_newest_sigma_and_mu(tau_tilde, nu_tilde, f_var_pre,
                                                                f_mu_pre)

        K = f_var_pre
        # ^ we call Sigma_0 K to match notation in GPML because very often the initial Sigma will be
        # the covariance matrix.

        # Recalculate terms that will be used often (based on latest sigma):
        tau_mi, nu_mi = self._create_cavity_dist(Sigma_final, mu_final, tau_tilde, nu_tilde)
        Sigma_sq_mi = tf.reciprocal(tau_mi, name="Sigma_sq_mi")
        mu_mi = tf.multiply(nu_mi, Sigma_sq_mi, name="mu_mi")
        chol_b, _ = self._compute_chol_B_and_S_tilde_sqrt_Sigma_0(tau_tilde, K)

        # Eqn 3.73 of GPML
        term_1_and_4 = tf.identity(
            0.5 * tf.reduce_sum(tf.log1p(tau_tilde * Sigma_sq_mi), name="sum_log1p_tau_tilde_sigma") - \
            tf.reduce_sum(tf.log(tf.diag_part(chol_b)), name="summed_log_chol_b_diag"), name="term_1_and_4")

        # Some general terms useful for all equations.
        Sigma_tilde = tf.diag(tf.squeeze(tf.reciprocal(tau_tilde)), name="Sigma_tilde")
        S_tilde = tf.diag(tf.squeeze(tau_tilde), name="Stilde")
        T = tf.diag(tf.squeeze(tau_mi), name="T")
        T_plus_S_tilde_inv = tf.diag(tf.reciprocal(tf.diag_part(S_tilde + T)),
                                     name="T_plus_S_tilde_inv")
        # ^  inverse is just reciprocal as have diagonal matrix


        # Eqn 3.74 of GPML.
        Stilde_sqrt_K = tf.matmul(tf.sqrt(S_tilde), K, name="Stilde_sqrt_K_")
        Li_S_tilde_sqrt_K = tf.matrix_triangular_solve(chol_b, Stilde_sqrt_K, name="Li_S_tilde_sqrt_K")
        Li_S_tilde_sqrt_K_sq = tf.matmul(Li_S_tilde_sqrt_K, Li_S_tilde_sqrt_K, transpose_a=True,
                                         name="Li_S_tilde_sqrt_K_sq")
        bracketted_term = tf.identity(K - Li_S_tilde_sqrt_K_sq - T_plus_S_tilde_inv,
                                      name="bracketed_term")
        term_5a_and_2a = tf.identity(
            0.5 * tf.matmul(nu_tilde,tf.matmul(bracketted_term,nu_tilde), transpose_a=True),
                                     name="term_5a_and_2a")

        # the second term also has some extra terms when f_mu_pre is not zeros
        mean_all_zeros = tf.reduce_all(tf.equal(f_mu_pre, 0))

        def non_zero_pre_mean():
            K_plus_sigma_tilde = K + Sigma_tilde
            chol_K_plus_sigma_tilde = tf.cholesky(K_plus_sigma_tilde + settings.jitter * tf.eye(tf.shape(K)[0], dtype=settings.tf_float))
            intermed = tf.matrix_triangular_solve(chol_K_plus_sigma_tilde, f_mu_pre)
            term_2c = tf.identity(-0.5 * tf.matmul(intermed, intermed, transpose_a=True), name="term_2c")

            intermed2 = tf.matrix_triangular_solve(tf.transpose(chol_K_plus_sigma_tilde), intermed, lower=False)
            term_2b = tf.matmul(nu_tilde/tau_tilde, intermed2, transpose_a=True)
            return term_2c, term_2b

        def zero_pre_mean():
            return tf.constant(0., dtype=settings.tf_float), tf.constant(0., dtype=settings.tf_float)

        term_2c, term_2b = tf.cond(mean_all_zeros, zero_pre_mean, non_zero_pre_mean)


        # Eqn 3.75 of GPML
        right_half = tf.matmul(T_plus_S_tilde_inv,
                               (tf.matmul(S_tilde, mu_mi) - 2 * nu_tilde))
        term_5b = tf.identity(
            0.5 * tf.matmul(mu_mi, tf.matmul(T, right_half), transpose_a=True),
            name="term_5b")

        # Third term of eqn 3.65 of GPML. This one is the log zeroth moment one and
        # so still depends on the function we are approximating.
        term_3 = tf.reduce_sum(self.central_moments_calculator().calc_log_zero_moment(tau_mi, nu_mi))

        # Having defined all the terms we now add them up and are done!
        log_zep = tf.identity(term_1_and_4 + term_5a_and_2a + term_2b + term_2c + term_5b + term_3,
                              name="log_zep")
        return tf.squeeze(log_zep)  # remove extra dimensions.

    @classmethod
    @name_scope("compute_newest_sigma_and_mu")
    def compute_newest_sigma_and_mu(cls, tau_tilde, nu_tilde, Sigma_0, mu_0):
        """
        The basic formulae are
        Sigma_new = (Sigma_0^-1 + diag(taul_tilde))^-1    (1)
        mu_new = Sigma_new ( nu_0 + nu_tilde)             (2)

        Eqn (1) we will compute however in a more stable way using the Matrix Inversion Lemma.
        See GPML Section 3.6 for more details on this. We use a touch of jitter however when computing
        nu_0 for equation 2 to ensure that it is stable.

        Sigma0 2d tensor
        tau, nu and mu are column vectors.
        """
        Sigma_new = cls.compute_newest_sigma(tau_tilde, Sigma_0)

        # The mean:
        chol_Sigma_0 = tf.cholesky(
            Sigma_0 + settings.jitter * tf.eye(tf.shape(Sigma_0)[0], dtype=settings.tf_float))
        nu_0 = tf.cholesky_solve(chol_Sigma_0, mu_0, name="nu_0")
        # TODO: it may be the case that mu_0 and so by extension nu_0 is very often zero (eg when using
        # EP for binary GP classification on a zero mean function initial GP, in this case the Cholesky
        # will be computed in vain by TF. so perhaps use a TF cond to prevent this path being taken
        # needlessly. evaluate whether important first via profiling.

        nu_new = nu_0 + nu_tilde
        mu_new = tf.matmul(Sigma_new, nu_new)
        return Sigma_new, mu_new

    @classmethod
    def compute_newest_sigma(cls, tau_tilde, Sigma_0):
        chol_B, S_tilde_sqrt_Sigma_0 = cls._compute_chol_B_and_S_tilde_sqrt_Sigma_0(tau_tilde, Sigma_0)
        V = tf.matrix_triangular_solve(chol_B, S_tilde_sqrt_Sigma_0, lower=True)
        Sigma_new = tf.subtract(Sigma_0, tf.matmul(V, V, transpose_a=True), name="Sigma_new")
        return Sigma_new

    @classmethod
    def _compute_chol_B_and_S_tilde_sqrt_Sigma_0(cls, tau_tilde, Sigma_0):
        S_tilde_sqrt = tf.diag(tf.sqrt(tf.squeeze(tau_tilde)), name="S_tilde_sqrt")
        S_tilde_sqrt_Sigma_0 = tf.matmul(S_tilde_sqrt, Sigma_0)
        B = tf.eye(tf.shape(tau_tilde)[0], dtype=settings.tf_float) + tf.matmul(
            S_tilde_sqrt_Sigma_0, S_tilde_sqrt)
        chol_B = tf.cholesky(B, name="cholesky_b")
        return chol_B, S_tilde_sqrt_Sigma_0

    @classmethod
    @name_scope("calc_cavity_dist")
    def _create_cavity_dist(cls, sigma, mu, tau_tilde, nu_tilde):
        sigma_sq_i_inv = tf.reciprocal(tf.expand_dims(tf.diag_part(sigma), -1), name="sigma_sq_i_inv")
        tau_mi = tf.subtract(sigma_sq_i_inv, tau_tilde, name="tau_mi")
        tau_mi = tf.maximum(tau_mi, VERY_VERY_SMALL_NUMBER, "tau_mi_guaranteed_positive")
        # ^ we keep tau_mi positive as negative variance does not make sense. However consider
        # whether there is a more sensible way to deal with this problem...?
        nu_mi = tf.subtract(sigma_sq_i_inv * mu, nu_tilde, name="nu_mi")
        return tau_mi, nu_mi

    @classmethod
    @name_scope("update_tilde_params")
    def _update_tilde_params(cls, sigma_sq_hat, mu_hat, tau_mi, nu_mi):
        tau_hat = tf.reciprocal(sigma_sq_hat, name="tau_hat")
        tau_tilde_newest = tf.maximum(tf.identity(tau_hat - tau_mi, name="tau_tilde_newest"),
                                      VERY_VERY_SMALL_NUMBER, name="tau_tilde_new_positive_enforced")
        nu_tilde_newest = tf.identity(tau_hat * mu_hat - nu_mi, name="nu_tilde_newest")
        return tau_tilde_newest, nu_tilde_newest


def _create_mean_square_change(old, new, name=None):
    return tf.reduce_mean(tf.square(old - new), name=name)
