

import numpy as np
import tensorflow as tf

import abc
from typing import Tuple, TypeVar
from ..config import default_jitter, default_float
from ..mean_functions import Zero
from ..utilities import ops, positive, triangular
from .. import kullback_leiblers
from ..base import Parameter
from ..conditionals import conditional
from .util import inducingpoint_wrapper

from gpflow.models import BayesianModel

Data = TypeVar('Data', Tuple[tf.Tensor, tf.Tensor], tf.Tensor)
DataPoint = tf.Tensor
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]


class SVGPs(BayesianModel):

    def __init__(self,
                 kernels,
                 likelihood,
                 inducing_variables,
                 *,
                 mean_functions=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mus=None,
                 q_sqrts=None,
                 whiten: bool = True,
                 num_data=None):
        """
        - kernels, inducing_variables, mean_functions are appropriate
          lists of GPflow objects
        - likelihood is a GPflow object
        - num_latent is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__()
        self.num_latent = num_latent
        self.kernels = kernels
        self.num_components = len(kernels)
        if mean_functions is None:
            mean_functions = [Zero() for _ in range(self.num_components)]
        self.mean_functions = mean_functions
        self.likelihood = likelihood

        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten
        self.inducing_variables = \
            [inducingpoint_wrapper(inducing_variable) for inducing_variable in inducing_variables]

        # init variational parameters
        num_inducings = \
            [len(inducing_variable) for inducing_variable in inducing_variables]

        self._init_variational_parameters(num_inducings, q_mus, q_sqrts, q_diag)

    def _init_variational_parameters(self, num_inducings, q_mus, q_sqrts, q_diag):
        self.q_mus, self.q_sqrts = [], []

        for c in range(self.num_components):
            q_mu = None if q_mus is None else q_mus[c]
            q_sqrt = None if q_sqrts is None else q_sqrts[c]

            q_mu_, q_sqrt_ = \
                self._build_component_variational_parameters(num_inducings[c], q_mu, q_sqrt, q_diag)

            self.q_mus.append(q_mu_)
            self.q_sqrts.append(q_sqrt_)

    def _build_component_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
        """
        Constructs the mean and cholesky of the covariance of the variational Gaussian posterior.
        If a user passes values for `q_mu` and `q_sqrt` the routine checks if they have consistent
        and correct shapes. If a user does not specify any values for `q_mu` and `q_sqrt`, the routine
        initializes them, their shape depends on `num_inducing` and `q_diag`.

        Note: most often the comments refer to the number of observations (=output dimensions) with P,
        number of latent GPs with L, and number of inducing points M. Typically P equals L,
        but when certain multioutput kernels are used, this can change.

        Parameters
        ----------
        :param num_inducing: int
            Number of inducing variables, typically refered to as M.
        :param q_mu: np.array or None
            Mean of the variational Gaussian posterior. If None the function will initialise
            the mean with zeros. If not None, the shape of `q_mu` is checked.
        :param q_sqrt: np.array or None
            Cholesky of the covariance of the variational Gaussian posterior.
            If None the function will initialise `q_sqrt` with identity matrix.
            If not None, the shape of `q_sqrt` is checked, depending on `q_diag`.
        :param q_diag: bool
            Used to check if `q_mu` and `q_sqrt` have the correct shape or to
            construct them with the correct shape. If `q_diag` is true,
            `q_sqrt` is two dimensional and only holds the square root of the
            covariance diagonal elements. If False, `q_sqrt` is three dimensional.
        """

        q_mu = np.zeros((num_inducing, self.num_latent)) if q_mu is None else q_mu
        q_mu_ = Parameter(q_mu, dtype=default_float(), name="q_mu")  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent), dtype=default_float())
                q_sqrt_ = Parameter(ones, transform=positive(), name="q_sqrt")  # [M, P]
            else:
                q_sqrt = [np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent)]
                q_sqrt = np.array(q_sqrt)
                q_sqrt_ = Parameter(q_sqrt, transform=triangular(), name="q_sqrt")  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                q_sqrt_ = Parameter(q_sqrt, transform=positive(), name="q_sqrt")  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                q_sqrt_ = Parameter(q_sqrt, transform=triangular(), name="q_sqrt")  # [L|P, M, M]

        return q_mu_, q_sqrt_

    def prior_kls(self):
        return [kullback_leiblers.prior_kl(self.inducing_variables[c],
                                           self.kernels[c],
                                           self.q_mus[c],
                                           self.q_sqrts[c],
                                           whiten=self.whiten) for c in range(self.num_components)]

    def prior_kl(self):
        return tf.reduce_sum(self.prior_kls())

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()
        f_means, f_vars = self.predict_fs(X, full_cov=False, full_output_cov=False)

        # hard coded additive model
        f_mean = tf.reduce_sum(f_means, axis=-1)[..., None]
        f_var = tf.reduce_sum(f_vars, axis=-1)[..., None]

        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)

        return tf.reduce_sum(var_exp) - kl

    def elbo(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.log_marginal_likelihood(data)

    def predict_fs(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:
        mus, vars = [], []
        for c in range(self.num_components):
            # hard coding ordering
            Xnew_c = Xnew[:, c:c+1]
            mu, var = conditional(Xnew_c,
                                  self.inducing_variables[c],
                                  self.kernels[c],
                                  self.q_mus[c],
                                  q_sqrt=self.q_sqrts[c],
                                  full_cov=full_cov,
                                  white=self.whiten,
                                  full_output_cov=full_output_cov)
            mu += self.mean_functions[c](Xnew_c)

            mus.append(mu)
            vars.append(var)

        return tf.concat(mus, axis=-1), tf.concat(vars, axis=-1)


class CNY(SVGPs):

    def __init__(self,
                 kernels,
                 likelihood,
                 inducing_variables,
                 *,
                 mean_functions=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mus=None,
                 q_sqrts=None,
                 whiten: bool = True,
                 num_data=None,
                 new_years=None):

        self.new_years = new_years
        self.num_years = len(new_years)

        num_latent = 1

        super().__init__(kernels,  likelihood, inducing_variables,
                         mean_functions=mean_functions,
                         num_latent=num_latent,
                         q_diag=q_diag,
                         q_mus=q_mus,
                         q_sqrts=q_sqrts,
                         whiten=whiten,
                         num_data=num_data)

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()


        f_means, f_vars = self.predict_fs(X, full_cov=False, full_output_cov=False)
        f_trend_mean = f_means[:, 0:1]
        f_trend_var = f_vars[:, 0:1]

        var_exp = 0
        for ny in range(self.num_years):
            X_ = X - self.new_years[ny]
            f_means, f_vars = self.predict_fs(X_, full_cov=False, full_output_cov=False)
            f_ny_mean = f_means[:, 1:2]
            f_ny_var = f_vars[:, 1:2]

            # hard coded ny model
            f_mean = f_trend_mean + f_ny_mean
            f_var = f_trend_var + f_ny_var

            var_exp += tf.reduce_sum(
                self.likelihood.variational_expectations(f_mean, f_var, Y[:, ny:ny+1])
            )

        return var_exp - kl

    def predict_years(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:


        f_means, f_vars = self.predict_fs(Xnew, full_cov=False, full_output_cov=False)
        f_trend_mean = f_means[:, 0:1]
        f_trend_var = f_vars[:, 0:1]

        var_exp = 0

        f_means_year = []
        f_vars_year = []
        for ny in range(self.num_years):
            X_ = Xnew - self.new_years[ny]
            f_means, f_vars = self.predict_fs(X_, full_cov=False, full_output_cov=False)
            f_ny_mean = f_means[:, 1:2]
            f_ny_var = f_vars[:, 1:2]

            # hard coded ny model
            f_mean = f_trend_mean + f_ny_mean
            f_var = f_trend_var + f_ny_var

            f_vars_year.append(f_var)
            f_means_year.append(f_mean)

        return tf.concat(f_means_year, axis=-1), tf.concat(f_vars_year, axis=-1)
