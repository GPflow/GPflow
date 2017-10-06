

from __future__ import absolute_import
import collections


import tensorflow as tf
import numpy as np

from .model import GPModel
from ..params.parameter import Parameter
from ..core.tensor_converter import TensorConverter
from .. import likelihoods
from .. import settings
from .. import mean_functions
from ..decors import params_as_tensors
from gpflow.decors import autoflow


float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

VERY_SMALL_NUMBER = 1e-6

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


    def __init__(self, X, Y, kern, name='name'):
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

        self.max_ep_steps = 5
        self.convergent_tol = 1e-4


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

                # Have the old ones be moved far enough away from the new one to make sure that we
                # go through the conditions initially
                nu_tilde_old=10 * self.convergent_tol * tf.ones_like(self.nu_tilde, dtype=float_type),
                tau_tilde_old=10 * self.convergent_tol * tf.ones_like(self.nu_tilde, dtype=float_type),

                # See GPML Algo 3.5 for good initial starting values for Sigma and mu
                sigma=K, mu=tf.zeros_like(self.nu_tilde, dtype=float_type),

                chol_b=tf.eye(tf.shape(self.Y)[0], dtype=float_type),

                counter=tf.constant(0)
        )

        std_normal_dist = _create_std_normal()

        def condition(ep_iter_vals):

            nu_change = tf.reduce_max(tf.abs(ep_iter_vals.nu_tilde_new - ep_iter_vals.nu_tilde_old))
            #tau_change = tf.reduce_max(tf.abs(ep_iter_vals.tau_tilde_new - ep_iter_vals.tau_tilde_old))
            #converged = tf.logical_and(tf.less_equal(nu_change, self.convergent_tol),
            #                           tf.less_equal(tau_change, self.convergent_tol))
            converged = tf.less_equal(nu_change, self.convergent_tol)
            # We carry on if we have not converged and we are less than the maximum number of steps.
            return tf.logical_and(tf.less(ep_iter_vals.counter, self.max_ep_steps),
                                  tf.logical_not(converged))


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
            zi = tf.identity(Y_between_m1_and_1 * mu_mi / sqrt_one_plus_sigma_sq_mi, name="zi")
            norm_pdf_over_cdf_for_zi = tf.divide(std_normal_dist.prob(zi, name="prob_zi"), (VERY_SMALL_NUMBER + std_normal_dist.cdf(zi, name="cdf_zi")), name="norm_pdf_over_cdf_for_zi")
            #TODO more stable way to do this^^^^^ look at other libraries.
            mu_hat_i = tf.add(mu_mi, (Y_between_m1_and_1 * sigma_sq_mi * norm_pdf_over_cdf_for_zi / sqrt_one_plus_sigma_sq_mi),
                              name="mu_hat_i")
            sigma_sq_hat_i = tf.subtract(sigma_sq_mi, (tf.square(sigma_sq_mi) * norm_pdf_over_cdf_for_zi / (1 + sigma_sq_mi)) * \
                                           (zi + norm_pdf_over_cdf_for_zi), name="sigma_sq_hat_i")

            # Lines 8-9 of GPML Algo 3.5
            tau_hat = tf.reciprocal(sigma_sq_hat_i)
            tau_tilde_newest = tf.maximum(tf.identity(tau_hat - tau_mi, name="tau_tilde_newest"), VERY_SMALL_NUMBER, name="tau_tilde_new_positive_enforced")
            nu_tilde_newest = tf.identity(tau_hat * mu_hat_i - nu_mi, name="nu_tilde_newest")

            # Lines 13-15 of GPML Algo 3.5 (as doing all at once do not bother with rank one updates)
            L, S_half_K = _cholesky_b(tau_tilde_newest, K, num_data=tf.shape(self.Y)[0])
            V = tf.matrix_triangular_solve(tf.transpose(L), S_half_K, lower=False)
            Sigma_newest = tf.subtract(K, tf.matmul(V, V, transpose_a=True), name="Sigma_new")
            mu_newest = tf.matmul(Sigma_newest, nu_tilde_newest, name="mu_new")

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
        #TODO: remove need of hack by breaking this method up into multiple functions.
        setattr(self, TensorConverter.__tensor_mode__, None)
        update_ops = [self.tau_tilde.unconstrained_tensor.assign(final_params.tau_tilde_new),
                      self.nu_tilde.unconstrained_tensor.assign(final_params.nu_tilde_new)]
        setattr(self, TensorConverter.__tensor_mode__, True)



        return_results = self.EPResults(nu_tilde=tf.stop_gradient(final_params.nu_tilde_new),
                               tau_tilde=tf.stop_gradient(final_params.tau_tilde_new),
                               sigma=tf.stop_gradient(final_params.sigma),
                               mu=tf.stop_gradient(final_params.mu),
                               chol_b=tf.stop_gradient(final_params.chol_b),
                               num_iter=tf.stop_gradient(final_params.counter))

        return return_results, update_ops


    @params_as_tensors
    def _build_likelihood(self):
        results, update_ops = self._run_ep()

        # Recalculate these based on latest sigma.
        sigma_sq_i_inv, tau_mi, nu_mi = _calc_tau_mi(results.sigma,
                                                     results.tau_tilde, results.mu,
                                                     results.nu_tilde)
        sigma_sq_mi = tf.reciprocal(tau_mi)
        mu_mi = nu_mi * sigma_sq_mi

        # Eqn 3.73 of GPML
        term_one_and_four = 0.5 * tf.reduce_sum(tf.log1p(results.tau_tilde/tau_mi)) - \
            tf.reduce_sum(tf.log(tf.diag_part(results.chol_b)))

        # Some general terms useful for all equations.
        S_tilde = tf.diag(tf.squeeze(results.tau_tilde), name="Stilde")
        T = tf.diag(tf.squeeze(tau_mi))
        T_p_S_tilde_inv = tf.reciprocal(S_tilde + T)  # inverse is just reciprocal as have diagonal matrix
        K = self.kern.K(self.X)

        # Eqn 3.74 of GPML.
        Stilde_sqrt_K = tf.matmul(tf.sqrt(S_tilde), K, name="Stilde_sqrt_K_")
        Li_S_tilde_sqrt_K = tf.matrix_triangular_solve(results.chol_b, Stilde_sqrt_K)
        Li_S_tilde_sqrt_K_sq = tf.matmul(Li_S_tilde_sqrt_K, Li_S_tilde_sqrt_K, transpose_a=True)
        bracketted_term = K - Li_S_tilde_sqrt_K_sq - T_p_S_tilde_inv
        first_half_term_five_and_second = 0.5 * tf.matmul(results.nu_tilde,
                                    tf.matmul(bracketted_term, results.nu_tilde), transpose_a=True)

        # Eqn 3.75 of GPML
        right_half = tf.matmul(T_p_S_tilde_inv, (tf.matmul(S_tilde, mu_mi) - 2 * results.nu_tilde))
        second_half_fifth_term = 0.5 * tf.matmul(mu_mi, tf.matmul(T, right_half), transpose_a=True)

        # Third term of eqn 3.65 of GPML
        std_normal = _create_std_normal()
        third_term = tf.reduce_sum(std_normal.log_cdf((self._switch_targets_to_minus_one_one(self.Y) *mu_mi) / tf.sqrt(1 + sigma_sq_mi)))

        # We update our store of the parameters to try to ensure faster convergence next time.
        with tf.control_dependencies(update_ops):
            log_likelihood = term_one_and_four + first_half_term_five_and_second + \
                             second_half_fifth_term + third_term
        return log_likelihood



    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        # Algorithm 3.6 of GPML.
        # SHOULD call run ep after any changes to hyperparameters.
        #TODO: there is some computation here that could get cached.
        #TODO Consider whether we can reuse any existing code.
        num_data = tf.shape(self.Y)[0]
        S_tilde_half = tf.diag(tf.squeeze(self.tau_tilde))
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
        #fixme convert back out into new space
        return f_new, var

    @autoflow()
    def run_ep(self):
        """
        Runs the EP and updates the natural parameters. Should be used before doing predictions.
        To cache some of the work. Unfortunatley at the moment we do not have any code to work out
        when the EP parameters are stale and need to be recomputed.
        """
        return_res, update_ops = self._run_ep()

        # Get these assigned to the resepective params objects.
        with tf.control_dependencies(update_ops):
            tau_tilde = tf.identity(return_res.tau_tilde, name="tau_tilde_final")
            nu_tilde = tf.identity(return_res.nu_tilde, name="nu_tilde_final")
            num_iters = tf.identity(return_res.num_iter, name="ep_iters")

        return tau_tilde, nu_tilde, num_iters

    def _switch_targets_to_minus_one_one(self, targets):
        return 2 * targets - 1  # matches the scale of targets in GPML. ie between -1 and 1.



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
        tau_mi = tf.maximum(tf.subtract(sigma_sq_i_inv, tau_tilde, name="tau_mi"), VERY_SMALL_NUMBER, name="tau_mi_limited_to_zero")
        #FIXME --  can tau_mi actually be negative? is this a good thing to be doing?
        nu_mi = tf.subtract(sigma_sq_i_inv * mu, nu_tilde, name="nu_mi")
    return sigma_sq_i_inv, tau_mi, nu_mi


def _normal_cdf(x):
    return (1. + tf.erf(x / np.sqrt(2.))) / 2.

def _create_std_normal():
    return tf.distributions.Normal(np_float_type(0.), np_float_type(1.))