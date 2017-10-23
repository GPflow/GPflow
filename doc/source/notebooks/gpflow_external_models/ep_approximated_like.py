

import enum

import tensorflow as tf
import numpy as np

from gpflow.models.model import GPModel
from gpflow.params.parameter import Parameter
from gpflow import settings
from gpflow import mean_functions
from gpflow.decors import autoflow
from gpflow import inference_helpers
from gpflow import decors

DEFAULT_MAX_ITER = 100
DEFAULT_CONVERGENT_TOL = 1e-6


class EPForLikelihood(enum.Enum):
    EVERYTIME = 1
    USE_CACHE = 2


class EPLikeApproxGP(GPModel):
    """
    Gaussian Processes where the effect of the likelihood is
    approximated via Expectation Propagation. This means that the `return_central_moment_calculator`
    method of the likelihood class should be implemented.

    When using the class autoflow functions such as predict_f this class does not recalculate the
    EP parameters for you. That means after changing hyperparameters (including just implicitly
    via optimization) you should call the run_ep method to update these params. This class will
    run EP on each _build_likelihood run unless you set `use_cache_on_like` False in the
    constructor.

    Currently only works on one latent factor GP.

    If you were to run this for GP Binary classification using a Gaussian CDF likelihood (ie a
    probit invlink) then a good reference is Section 3.6 of
    ::
        @book{rasmussen2006gaussian,
          title={Gaussian processes for machine learning},
          author={Rasmussen, Carl Edward and Williams, Christopher KI},
          volume={1},
          year={2006},
          publisher={MIT press Cambridge}
        }

    A good summary of uses of EP can be found at https://tminka.github.io/papers/ep/roadmap.html.

    Warning: this class lives outside the main codebase and so is untested.
    """
    def __init__(self, X, Y, kern, likelihood,  name='name', use_cache_on_like=False,
                 num_latent=None, ep_runner=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x 1. It is labels of the classes. 0 and 1s.
        - kern is appropriate GPflow objects
        - use_cache_on_like: will mean that you do not run EP each time you evaluate the build
        likelihood. (ie the EP run will not be connected up to the TF graph). If you set to use the
        cache then it will use the old (possibly stale) EP params.
        - ep runner is the class that will run EP. It should be a
        gpflow.inference_helpers.ep_runner.EPRunner class.
        """

        # Currently only supports zero mean function. This allows us to use a neat and simple build
        # predict function.
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function=mean_functions.Zero(),
                         name=name)

        self.num_latent = num_latent or Y.shape[1]
        assert self.num_latent == 1, "Only works for one latent factor GP"

        self.nu_tilde = Parameter(np.zeros((Y.shape[0], self.num_latent), dtype=settings.np_float),
                                  trainable=False, name="nu_tilde")
        self.tau_tilde = Parameter(np.zeros((Y.shape[0], self.num_latent), dtype=settings.np_float),
                                  trainable=False, name="tau_tilde")

        if ep_runner is None:
            self.ep_runner = inference_helpers.EPRunner(
                lambda : self.likelihood.return_central_moment_calculator(self.Y),
                max_iter=DEFAULT_MAX_ITER, convergent_tol=DEFAULT_CONVERGENT_TOL)

        self._run_ep_for_likelihood_flag = EPForLikelihood.USE_CACHE if use_cache_on_like else EPForLikelihood.EVERYTIME

    def clear_ep_params(self, session=None):
        """
        Convenience function to reset both EP params back to zero.
        Clears the EP parameters so that next time we start from zeros for them. This sometimes
        helps convergence if we move far away from where we had trained the solution before.
        """
        self.nu_tilde.assign(
            np.zeros_like(self.nu_tilde.read_value(session), dtype=settings.np_float), session
        )
        self.tau_tilde.assign(
            np.zeros_like(self.tau_tilde.read_value(session), dtype=settings.np_float), session
        )

    @autoflow()
    def run_ep(self):
        """
        Runs the EP and updates the natural parameters. Should be used before doing predictions.
        To cache some of the work. Unfortunatley at the moment we do not have any code to work out
        when the EP parameters are stale and need to be recomputed.
        """
        ep_params, update_ops = self._run_ep()

        # Get these assigned to the respective params objects.
        with tf.control_dependencies(update_ops):
            tau_tilde = tf.identity(ep_params.new_tau_tilde, name="tau_tilde_final")
            nu_tilde = tf.identity(ep_params.new_nu_tilde, name="nu_tilde_final")
            num_iters = tf.identity(ep_params.counter, name="ep_iters")

        return tau_tilde, nu_tilde, num_iters

    @decors.params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        # Here we use the fact that the prior sigma is K and that the prior mean is also zero. We
        # therefore can have a simple expression with stable inverses. If not then we could compute
        # the new mean and covariance and use conditionals.py

        # See GPML Algorithm 3.6 for the suggested stable way to compute mu and var implemented below.
        num_data = tf.shape(self.Y)[0]
        S_tilde_half = tf.diag(tf.sqrt(tf.squeeze(self.tau_tilde)))
        Kmm = self.kern.K(self.X)
        B = tf.eye(num_data, dtype=settings.tf_float) + tf.matmul(S_tilde_half, tf.matmul(Kmm, S_tilde_half))
        L = tf.cholesky(B)
        z = tf.matmul(S_tilde_half, tf.cholesky_solve(L,
                                                      tf.matmul(S_tilde_half, tf.matmul(Kmm, self.nu_tilde))))
        Kmn = self.kern.K(self.X, Xnew)
        f_new = tf.matmul(Kmn, (self.nu_tilde - z), transpose_a=True)

        v = tf.matrix_triangular_solve(L, tf.matmul(S_tilde_half, Kmn))

        if full_cov:
            Knn = self.kern.K(Xnew)
            var = Knn - tf.matmul(v, v, transpose_a=True)
        else:
            Knn = self.kern.Kdiag(Xnew)
            var = Knn - tf.reduce_sum(tf.square(v), axis=0)

        var = tf.expand_dims(var, -1, name="pad")
        # ^ pad with one extra dim to reflect the one latent factor that we condition on.
        return f_new, var

    @decors.params_as_tensors
    def _build_likelihood(self):
        if self._run_ep_for_likelihood_flag is EPForLikelihood.USE_CACHE:
            update_ops = [tf.no_op(name="no_ep_update")]
        elif self._run_ep_for_likelihood_flag is EPForLikelihood.EVERYTIME:
            with tf.name_scope("updating_ep_for_ll"):
                ep_results, update_ops = self._run_ep()
        else:
            raise NotImplementedError(
                "Unsupported method: {}".format(self.run_ep_for_likelihood_flag.name))
        K = self.kern.K(self.X)
        zero_mean = tf.zeros_like(self.nu_tilde)
        with tf.control_dependencies(update_ops), decors.params_as_tensors_for(self, convert=False):
                tau_tilde = self.tau_tilde.unconstrained_tensor.read_value()
                nu_tilde = self.nu_tilde.unconstrained_tensor.read_value()

        log_z = self.ep_runner.calculate_log_normalising_constant(zero_mean, K,
                                        nu_tilde, tau_tilde)
        return log_z

    @decors.params_as_tensors
    def _run_ep(self):
        K = self.kern.K(self.X)
        f_mu_pre = tf.zeros_like(self.nu_tilde)
        ep_params = self.ep_runner.run_ep(f_mu_pre, K + 1e-6*tf.eye(tf.shape(K)[0], dtype=settings.tf_float) , self.nu_tilde, self.tau_tilde)
        with decors.params_as_tensors_for(self, convert=False):
            update_ops = [
                self.nu_tilde.unconstrained_tensor.assign(ep_params.new_nu_tilde),
                self.tau_tilde.unconstrained_tensor.assign(ep_params.new_tau_tilde)
            ]
        return ep_params, update_ops
