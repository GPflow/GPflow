

from __future__ import absolute_import
import collections
import enum
import contextlib


import tensorflow as tf
import numpy as np

from .model import GPModel
from ..params.parameter import Parameter
from ..core.tensor_converter import TensorConverter
from .. import likelihoods
from .. import settings
from .. import mean_functions
from ..decors import params_as_tensors
from ..decors import params_as_tensors_for
from gpflow.decors import autoflow
from .. import gaussian_utils


float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

VERY_VERY_SMALL_NUMBER = 1e-20


class EPForLikelihood(enum.Enum):
    EVERYTIME = 1
    USE_CACHE = 2


class EPBinClassGP(GPModel):
    """
    Gaussian Process Binary Classification via Expectation Propagation.


    The reference for this is Section 3.6 of
    ::
        @book{rasmussen2006gaussian,
          title={Gaussian processes for machine learning},
          author={Rasmussen, Carl Edward and Williams, Christopher KI},
          volume={1},
          year={2006},
          publisher={MIT press Cambridge}
        }
    """

    EPResults = collections.namedtuple("EPResults", 'nu_tilde, tau_tilde, sigma, mu, chol_b, num_iter')


    def __init__(self, X, Y, kern, name='name', use_cache_on_like=True):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x 1. It is labels of the classes. 0 and 1s.
        kern is appropriate GPflow objects
        """
        likelihood = likelihoods.Bernoulli(invlink=likelihoods.probit)

        GPModel.__init__(self, X, Y, kern, likelihood, mean_function=mean_functions.Zero(),
                         name=name)

        self.num_latent = Y.shape[1]
        assert self.num_latent == 1, "Only works for Binary Classification"

        self.nu_tilde = Parameter(np.zeros(Y.shape, dtype=np_float_type), trainable=False,
                                  name="nu_tilde")
        self.tau_tilde = Parameter(np.zeros(Y.shape, dtype=np_float_type), trainable=False,
                                  name="tau_tilde")

        self.max_ep_steps = 100
        self.convergent_tol = 1e-6
        self.run_ep_for_likelihood_flag = EPForLikelihood.USE_CACHE if use_cache_on_like else EPForLikelihood.EVERYTIME


    @params_as_tensors
    def _run_ep(self):
        K = tf.stop_gradient(self.kern.K(self.X), name="K_stop_grad")
        # ^ we do not want gradients to go back through this route. They should be zero anyway but
        # just to make sure.
        Y_between_m1_and_1 = self._switch_targets_to_minus_one_one(self.Y)


        EPIterVars = collections.namedtuple("EPIterVars",
                                            'nu_tilde_new, tau_tilde_new, nu_tilde_old, tau_tilde_old, sigma, mu, chol_b, counter')
        initial_vars = EPIterVars(
                # Start the current params from their last value to help convergence be faster
                nu_tilde_new=self.nu_tilde, tau_tilde_new=self.tau_tilde,

                nu_tilde_old=tf.ones_like(self.nu_tilde, dtype=float_type),
                tau_tilde_old=tf.ones_like(self.nu_tilde, dtype=float_type),

                # See GPML Algo 3.5 for good initial starting values for Sigma and mu
                sigma=K + 1e-6*tf.eye(tf.shape(K)[0], dtype=float_type),
                mu=tf.zeros_like(self.nu_tilde, dtype=float_type),

                chol_b=tf.eye(tf.shape(self.Y)[0], dtype=float_type),

                counter=tf.constant(0)
        )

        std_normal_dist = _create_std_normal()

        def condition(ep_iter_vals):

            nu_change = tf.reduce_mean(tf.square(ep_iter_vals.nu_tilde_new - ep_iter_vals.nu_tilde_old), name="mean_nu_change")
            tau_change = tf.reduce_mean(tf.square(ep_iter_vals.tau_tilde_new - ep_iter_vals.tau_tilde_old), name="mean_tau_change")
            converged = tf.logical_and(tf.less_equal(nu_change, self.convergent_tol),
                                       tf.less_equal(tau_change, self.convergent_tol), name="converged_bool")
            #converged = tf.less_equal(nu_change, self.convergent_tol)
            # We carry on if we have not converged and we are less than the maximum number of steps.
            return tf.logical_or(tf.less(ep_iter_vals.counter, 1), # ensure run at least once.
                                 tf.logical_and(tf.less(ep_iter_vals.counter, self.max_ep_steps),
                                               tf.logical_not(converged)))


        def body(ep_iter_vals):
            # We do all of the cavities in parallel.
            # lines 4-6 of GPML: Algo 3.5
            sigma_sq_i_inv, tau_mi, nu_mi = _calc_tau_mi(ep_iter_vals.sigma,
                                                         ep_iter_vals.tau_tilde_new, ep_iter_vals.mu,
                                                         ep_iter_vals.nu_tilde_new)

            # Line 7 of GPML Algo 3.5
            sigma_sq_mi = tf.reciprocal(tau_mi, name="sigma_sq_mi")
            mu_mi = tf.multiply(nu_mi, sigma_sq_mi, name="mu_mi")
            sqrt_one_plus_sigma_sq_mi = tf.sqrt(1 + sigma_sq_mi, name="sqrt_one_plus_sigma_sq_mi")

            tau_mi_plus_tau_mi_sq = tau_mi + tf.square(tau_mi)

            zi = tf.identity(Y_between_m1_and_1 * mu_mi / sqrt_one_plus_sigma_sq_mi, name="zi")
            norm_pdf_over_cdf_for_zi = gaussian_utils.deriv_log_cdf_normal(zi)
            #norm_pdf_over_cdf_for_zi = std_normal_dist.prob(zi) / (std_normal_dist.cdf(zi) + 1e-15)

            mu_hat_i = tf.add(mu_mi, (Y_between_m1_and_1 * norm_pdf_over_cdf_for_zi / tf.sqrt(tau_mi_plus_tau_mi_sq)),
                              name="mu_hat_i")
            sigma_sq_hat_i = tf.subtract(sigma_sq_mi, (norm_pdf_over_cdf_for_zi / tau_mi_plus_tau_mi_sq) * \
                                           (zi + norm_pdf_over_cdf_for_zi), name="sigma_sq_hat_i")

            # Lines 8-9 of GPML Algo 3.5
            tau_hat = tf.reciprocal(sigma_sq_hat_i, name="tau_hat")
            tau_tilde_newest = tf.maximum(tf.identity(tau_hat - tau_mi, name="tau_tilde_newest"), VERY_VERY_SMALL_NUMBER, name="tau_tilde_new_positive_enforced")
            nu_tilde_newest = tf.identity(tau_hat * mu_hat_i - nu_mi, name="nu_tilde_newest")

            #(as doing all at once do not bother with rank one updates)
            Sigma_newest, mu_newest, L = self._compute_sigma_and_mu_newest(tau_tilde_newest, nu_tilde_newest, K)

            # Pack everything up for next time around.
            new_iter_vals = EPIterVars(
                nu_tilde_new=nu_tilde_newest, tau_tilde_new=tau_tilde_newest, nu_tilde_old=ep_iter_vals.nu_tilde_new,
                tau_tilde_old=ep_iter_vals.tau_tilde_new, sigma=Sigma_newest, mu=mu_newest,
                chol_b=L, counter=tf.identity(ep_iter_vals.counter + 1, name="counter_add")
            )
            return (new_iter_vals,)

        final_params = tf.while_loop(cond=condition, body=body, loop_vars=(initial_vars,), back_prop=False)[0]


        # A bit of a hack but I want to get hold of the Parameter objects and set the unconstrained
        # tensor -- which is the actual variable that I can assign to.
        #TODO: remove need of hack by breaking this method up into multiple functions...?
        with params_as_tensors_for(self, convert=False):
            update_ops = [self.tau_tilde.unconstrained_tensor.assign(final_params.tau_tilde_new),
                          self.nu_tilde.unconstrained_tensor.assign(final_params.nu_tilde_new)]


        # I'm stopping all the gradients below. Im not quite sure how necessary this is
        # as I put back prop on the while loop to be False.
        # Note that the final sigma matrix depends on K so you do need to rerun
        # with allowing grads if want to actually optimise.
        return_results = self.EPResults(nu_tilde=tf.stop_gradient(final_params.nu_tilde_new),
                               tau_tilde=tf.stop_gradient(final_params.tau_tilde_new),
                               sigma=tf.stop_gradient(final_params.sigma),
                               mu=tf.stop_gradient(final_params.mu),
                               chol_b=tf.stop_gradient(final_params.chol_b),
                               num_iter=tf.stop_gradient(final_params.counter))

        return return_results, update_ops


    @params_as_tensors
    def _build_likelihood(self):
        if self.run_ep_for_likelihood_flag is EPForLikelihood.USE_CACHE:
            update_ops = [tf.no_op(name="no_ep_update")]
            counter = -1

        elif self.run_ep_for_likelihood_flag is EPForLikelihood.EVERYTIME:
            with tf.name_scope("updating_ep_for_ll"):
                results_from_ep, update_ops = self._run_ep()
                counter = results_from_ep.num_iter
                # Dont want to use the results here as I stopped the gradient on the kernel, which
                # will be bad if you use this method for optimisation.
        else:
            raise NotImplementedError("Unsupported method: {}".format(self.run_ep_for_likelihood_flag.name))

        with tf.control_dependencies(update_ops):
            # compute the latest sigma and mu as it depends on the Kernel and so needs to be updated.
            Sigma_newest, mu_newest, L = self._compute_sigma_and_mu_newest(self.tau_tilde, self.nu_tilde, self.kern.K(self.X))

            with params_as_tensors_for(self, convert=False):
                results = self.EPResults(nu_tilde=self.nu_tilde.unconstrained_tensor.read_value(),
                                         tau_tilde=self.tau_tilde.unconstrained_tensor.read_value(),
                                         sigma=Sigma_newest,
                                     mu=mu_newest, chol_b=L, num_iter=counter)
            # ^ use read_value to force TF to use the assigned variables if for some reason we have
            # done the assign on another device.


        # Recalculate these based on latest sigma:
        sigma_sq_i_inv, tau_mi, nu_mi = _calc_tau_mi(results.sigma,
                                                     results.tau_tilde, results.mu,
                                                     results.nu_tilde)
        sigma_sq_mi = tf.reciprocal(tau_mi)
        mu_mi = nu_mi * sigma_sq_mi

        # Eqn 3.73 of GPML
        term_one_and_four = tf.identity(0.5 * tf.reduce_sum(tf.log1p(results.tau_tilde * sigma_sq_mi)) - \
            tf.reduce_sum(tf.log(tf.diag_part(results.chol_b))), name="term_one_and_four")

        # Some general terms useful for all equations.
        S_tilde = tf.diag(tf.squeeze(results.tau_tilde), name="Stilde")
        T = tf.diag(tf.squeeze(tau_mi), name="T")
        T_plus_S_tilde_inv = tf.diag(tf.reciprocal(tf.diag_part(S_tilde + T)), name="T_plus_S_tilde_inv")
        # ^  inverse is just reciprocal as have diagonal matrix
        K = tf.identity(self.kern.K(self.X), name='Kernel_at_x')

        # Eqn 3.74 of GPML.
        Stilde_sqrt_K = tf.matmul(tf.sqrt(S_tilde), K, name="Stilde_sqrt_K_")
        Li_S_tilde_sqrt_K = tf.matrix_triangular_solve(results.chol_b, Stilde_sqrt_K)
        Li_S_tilde_sqrt_K_sq = tf.matmul(Li_S_tilde_sqrt_K, Li_S_tilde_sqrt_K, transpose_a=True)
        bracketted_term = K - Li_S_tilde_sqrt_K_sq - T_plus_S_tilde_inv
        first_half_term_five_and_second = tf.identity(0.5 * tf.matmul(results.nu_tilde,
                                    tf.matmul(bracketted_term, results.nu_tilde), transpose_a=True),
                                                      name="first_half_term_five_and_second")

        # Eqn 3.75 of GPML
        right_half = tf.matmul(T_plus_S_tilde_inv, (tf.matmul(S_tilde, mu_mi) - 2 * results.nu_tilde))
        second_half_fifth_term = tf.identity(0.5 * tf.matmul(mu_mi, tf.matmul(T, right_half), transpose_a=True),
                                             name="second_half_fifth_term")

        # Third term of eqn 3.65 of GPML
        std_normal = _create_std_normal()
        third_term = tf.reduce_sum(std_normal.log_cdf((self._switch_targets_to_minus_one_one(self.Y) *mu_mi) / tf.sqrt(1 + sigma_sq_mi)),
                                   name="third_term")


        log_likelihood = term_one_and_four + first_half_term_five_and_second + \
                             second_half_fifth_term + third_term
        return tf.identity(tf.squeeze(log_likelihood), name="log_likelihood")



    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        # Algorithm 3.6 of GPML.
        # SHOULD call run ep after any changes to hyperparameters.
        #TODO: there is some computation here that could get cached.
        #TODO Consider whether we can reuse any existing code.
        num_data = tf.shape(self.Y)[0]
        S_tilde_half = tf.diag(tf.sqrt(tf.squeeze(self.tau_tilde)))
        Kmm = self.kern.K(self.X)
        B = tf.eye(num_data, dtype=float_type) + tf.matmul(S_tilde_half, tf.matmul(Kmm, S_tilde_half))
        L = tf.cholesky(B)

        z = tf.matmul(S_tilde_half, tf.cholesky_solve(L,
                                                      tf.matmul(S_tilde_half, tf.matmul(Kmm, self.nu_tilde))))
        Kmn = self.kern.K(self.X, Xnew)
        f_new = tf.matmul(Kmn, (self.nu_tilde - z), transpose_a=True)

        v = tf.matrix_triangular_solve(L, tf.matmul(S_tilde_half, Kmn))
        Knn = self.kern.K(Xnew)
        var = Knn - tf.matmul(v, v, transpose_a=True)

        if not full_cov:
            var = tf.expand_dims(tf.diag_part(var), -1)
            # ^ Consider whether faster way to do this.

        return f_new, var

    @autoflow()
    def run_ep(self):
        """
        Runs the EP and updates the natural parameters. Should be used before doing predictions.
        To cache some of the work. Unfortunatley at the moment we do not have any code to work out
        when the EP parameters are stale and need to be recomputed.
        """
        return_res, update_ops = self._run_ep()

        # Get these assigned to the respective params objects.
        with tf.control_dependencies(update_ops):
            tau_tilde = tf.identity(return_res.tau_tilde, name="tau_tilde_final")
            nu_tilde = tf.identity(return_res.nu_tilde, name="nu_tilde_final")
            num_iters = tf.identity(return_res.num_iter, name="ep_iters")

        return tau_tilde, nu_tilde, num_iters

    @staticmethod
    def _switch_targets_to_minus_one_one(targets):
        return 2 * targets - 1  # matches the scale of targets in GPML. ie between -1 and 1.

    def _compute_sigma_and_mu_newest(self, tau_tilde, nu_tilde, K):
        # Lines 13-15 of GPML Algo 3.5
        L, S_half_K = _cholesky_b(tau_tilde, K, num_data=tf.shape(self.Y)[0])
        V = tf.matrix_triangular_solve(L, S_half_K, lower=True, name="V")
        Sigma_newest = tf.subtract(K, tf.matmul(V, V, transpose_a=True), name="Sigma_new")
        mu_newest = tf.matmul(Sigma_newest, nu_tilde, name="mu_new")
        return Sigma_newest, mu_newest, L


def _cholesky_b(tau_tilde, K, num_data):
    with tf.name_scope("chol_b_calcs"):
        S_half = tf.diag(tf.sqrt(tf.squeeze(tau_tilde)), name="Stilde_half")
        S_half_K = tf.matmul(S_half, K, name="Stilde_half_K")
        Shalf_K_Shalf = tf.matmul(S_half_K, S_half, name="Shalf_K_Shalf")
        B = tf.identity(tf.eye(num_data, dtype=float_type) + Shalf_K_Shalf, name="B")
        chol_B = tf.cholesky(B, name="Cholesky_B")
    return chol_B, S_half_K


def _calc_tau_mi(sigma, tau_tilde, mu, nu_tilde):
    with tf.name_scope("calc_tau_mi"):
        sigma_sq_i_inv = tf.reciprocal(tf.expand_dims(tf.diag_part(sigma), -1), name="sigma_sq_i_inv")
        tau_mi = tf.maximum(tf.subtract(sigma_sq_i_inv, tau_tilde, name="tau_mi"), VERY_VERY_SMALL_NUMBER, name="tau_mi_limited_to_zero")
        #tau_mi = tf.subtract(sigma_sq_i_inv, tau_tilde, name="tau_mi")
        #TODO --  consider whether max is necessary here...
        nu_mi = tf.subtract(sigma_sq_i_inv * mu, nu_tilde, name="nu_mi")
    return sigma_sq_i_inv, tau_mi, nu_mi


def _normal_cdf(x):
    return (1. + tf.erf(x / np.sqrt(2.))) / 2.


def _create_std_normal():
    return tf.distributions.Normal(np_float_type(0.), np_float_type(1.))