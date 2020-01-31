# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
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

from typing import Tuple, TypeVar, List

import numpy as np
import tensorflow as tf

from .util import inducingpoint_wrapper
from .. import kullback_leiblers
from ..base import Module
from ..base import Parameter
from ..conditionals import conditional, base_conditional
from ..config import default_float, default_jitter
from ..mean_functions import Zero
from ..utilities import ops
from ..utilities import positive, triangular
from ..utilities.ops import eye

Data = TypeVar('Data', Tuple[tf.Tensor, tf.Tensor], tf.Tensor)
DataPoint = tf.Tensor
MeanAndVariance = Tuple[tf.Tensor, tf.Tensor]

from ..models.model import GPModel, BayesianModel

BlockDiag = tf.linalg.LinearOperatorBlockDiag
FullMatrix = tf.linalg.LinearOperatorFullMatrix

class SparseGP(Module):

    def __init__(self,
                 kernel,
                 inducing_variable,
                 *,
                 mean_function=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten: bool = True,
                 num_data=None,
                 offset=None):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
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
        if mean_function is None:
            mean_function = Zero()
        self.mean_function = mean_function
        self.kernel = kernel

        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten

        self.inducing_variable = inducingpoint_wrapper(inducing_variable)

        # init variational parameters
        self.num_inducing = len(self.inducing_variable)
        if offset is None:
            self.offset_x = None
            self.offset_y = None
        else:
            # assert offset.shape == (2, 1)
            self.offset_x = Parameter(offset[0])
            self.offset_y = Parameter(offset[1])

        self._init_variational_parameters(self.num_inducing, q_mu, q_sqrt, q_diag)


    @property
    def q_mu(self):
        if self.offset_x is None:
            return self._q_mu
        else:
            Kmm = self.kernel(self.inducing_variable.Z) + \
                  ops.eye(self.num_inducing, value=default_jitter(), dtype=default_float())
            Kmn = self.kernel(self.inducing_variable.Z, self.offset_x)
            Knn = self.kernel(self.offset_y, full=False)

            mu0, var = base_conditional(Kmn, Kmm, Knn, self._q_mu,
                                        full_cov=False, white=self.whiten, q_sqrt=self.q_sqrt)

            n, _ = base_conditional(Kmn, Kmm, Knn, tf.ones_like(self._q_mu),
                                    full_cov=False, white=self.whiten, q_sqrt=self.q_sqrt)
            return self._q_mu - ((mu0 - self.offset_y) / n)

    def _init_variational_parameters(self, num_inducing, q_mu, q_sqrt, q_diag):
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
        self._q_mu = Parameter(q_mu, dtype=default_float())  # [M, P]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent)]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                num_inducing = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def prior_kl(self):

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt

        return kullback_leiblers.prior_kl(self.inducing_variable,
                                          self.kernel,
                                          q_mu,
                                          q_sqrt,
                                          whiten=self.whiten)

    def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt

        mu, var = conditional(Xnew,
                              self.inducing_variable,
                              self.kernel,
                              q_mu,
                              q_sqrt=q_sqrt,
                              full_cov=full_cov,
                              white=self.whiten,
                              full_output_cov=full_output_cov)
        # tf.debugging.assert_positive(var)  # We really should make the tests pass with this here
        return mu + self.mean_function(Xnew), var

    def predict_f_samples(self,
                          predict_at: DataPoint,
                          num_samples: int = 1,
                          full_cov: bool = True,
                          full_output_cov: bool = False) -> tf.Tensor:
        """
        Produce samples from the posterior latent function(s) at the input points.
        """
        mu, var = self.predict_f(predict_at, full_cov=full_cov)  # [N, P], [P, N, N]
        num_latent = var.shape[0]
        num_elems = tf.shape(var)[1]
        var_jitter = ops.add_to_diagonal(var, default_jitter())
        L = tf.linalg.cholesky(var_jitter)  # [P, N, N]
        V = tf.random.normal([num_latent, num_elems, num_samples], dtype=mu.dtype)  # [P, N, S]
        LV = L @ V  # [P, N, S]
        mu_t = tf.linalg.adjoint(mu)  # [P, N]
        return tf.transpose(mu_t[..., np.newaxis] + LV)  # [S, N, P]


class MeanFieldSparseGPs(Module):

    def __init__(self,
                 kernels,
                 inducing_variables,
                 *,
                 mean_functions=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mus=None,
                 q_sqrts=None,
                 whiten: bool = True,
                 offsets_x=None,
                 offsets_y=None,
                 ):

        super().__init__()
        self.num_latent = num_latent
        self.kernels = kernels
        self.num_components = len(kernels)
        if mean_functions is None:
            mean_functions = [Zero() for _ in range(self.num_components)]
        self.mean_functions = mean_functions

        self.q_diag = q_diag
        self.whiten = whiten

        # init variational parameters
        num_inducings = \
            [len(inducing_variable) for inducing_variable in inducing_variables]
        self.num_inducings = num_inducings

        qs = []
        for c in range(self.num_components):
            q_mu = None if q_mus is None else q_mus[c]
            q_sqrt = None if q_sqrts is None else q_sqrts[c]
            offset_x = None if offsets_x is None else offsets_x[c]
            offset_y = None if offsets_y is None else offsets_y[c]
            qs.append(
                SparseGP(self.kernels[c], inducing_variables[c],
                         q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten, offset=[offset_x, offset_y]))
        self.qs = qs

    def predict_fs(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:
        mus, vars = [], []
        for q in self.qs:
            mu, var = q.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)
            mus.append(mu)
            vars.append(var)

        return tf.concat(mus, axis=-1), tf.concat(vars, axis=-1)


class SparseCoupledGPs(Module):

    def __init__(self,
                 kernels,
                 inducing_variables,
                 *,
                 mean_functions=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mus=None,
                 q_sqrts=None,
                 whiten: bool = True,
                 offsets_x=None,
                 offsets_y=None,
                 ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
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
        self.q_diag = q_diag
        self.whiten = whiten

        num_inducings = [len(iv) for iv in inducing_variables]
        self.num_inducings = num_inducings
        self.num_inducing = np.sum(self.num_inducings)

        inducing_variables = [inducingpoint_wrapper(iv) for iv in inducing_variables]
        self.inducing_variables = inducing_variables
        # init variational parameters
        self._init_variational_parameters(self.num_inducing, q_mus, q_sqrts, q_diag=self.q_diag)
        # initialize offsets

        if offsets_x is None:
            offsets_x = None
            offsets_y = None

        else:
            for c in range(self.num_components):
                offset_x = None if offsets_x is None else offsets_x[c]
                offsets_x[c] = Parameter(offset_x)
                offset_y = None if offsets_y is None else offsets_y[c]
                offsets_y[c] = Parameter(offset_y)
        # if offsets_y is None:
        #     offsets_y = None
        # else:
        #     offsets_y = Parameter(offsets_y)




        self.offsets_x = offsets_x
        self._offsets_y = offsets_y



    @property
    def offsets_y(self):
        return tf.concat([o.unconstrained_variable for o in self._offsets_y], axis=-1)


    @property
    def inducing_variable(self):
        return tf.concat([iv.Z for iv in self.inducing_variables], axis=0)

    @property
    def q_mus(self):
        return tf.concat(self._q_mus, axis=-1)

    def _init_variational_parameters(self, num_inducing, q_mus, q_sqrt, q_diag):
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

        if q_mus is None:
            self._q_mus = [
                Parameter(
                    np.zeros((self.num_inducings[c], self.num_latent)),
                    dtype=default_float())
                for c in range(self.num_components)
            ]

        if q_sqrt is None:
            if self.q_diag:
                ones = np.ones((num_inducing, self.num_latent), dtype=default_float())
                self.q_sqrt = Parameter(ones, transform=positive())  # [M, P]
            else:
                q_sqrt = [np.eye(num_inducing, dtype=default_float()) for _ in range(self.num_latent)]
                q_sqrt = np.array(q_sqrt)
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [P, M, M]
        else:
            if q_diag:
                assert q_sqrt.ndim == 2
                self.num_latent = q_sqrt.shape[1]
                self.q_sqrt = Parameter(q_sqrt, transform=positive())  # [M, L|P]
            else:
                assert q_sqrt.ndim == 3
                self.num_latent = q_sqrt.shape[0]
                self.q_sqrt = Parameter(q_sqrt, transform=triangular())  # [L|P, M, M]

    def K1(self, X: List):
        Ks = []
        for c in range(self.num_components):
            x = X[c].Z
            Ks.append(self.kernels[c].K(x))
        return BlockDiag([FullMatrix(K) for K in Ks])



    def K3(self, X: tf.Tensor):
        Ks = []
        for c in range(self.num_components):
            K = self.kernels[c].K_diag(X)[..., None, None]
            Ks.append(K)
        return BlockDiag([FullMatrix(K) for K in Ks])


    def K2(self, X: List, X2: tf.Tensor=None, full_cov=True):
        Ks = []
        for c in range(self.num_components):
            x = X[c].Z
            Ks.append(self.kernels[c].K(x, X2)[ None])

        zeros = [tf.zeros_like(K) for K in Ks]
        K = []
        for c1 in range(self.num_components):
            K.append([])
            for c2 in range(self.num_components):
                M = Ks[c1] if c1 == c2 else zeros[c2]
                K[-1].append(M)
        return tf.concat([tf.concat(row, axis=-3) for row in K], axis=-2)

    @property
    def q_mu(self):
        if self.offsets_x is None:
            return self._q_mu
        else:
            q_mus = []
            for c in range(self.num_components):

                kernel = self.kernels[c]
                Z = self.inducing_variables[c].Z
                offset_x = self.offsets_x[c]
                offset_y = 0. # self.offsets_y[c]
                num_inducing = self.num_inducings[c]
                q_mu = self._q_mus[c]

                Kmm = kernel(Z) + \
                      ops.eye(num_inducing, value=default_jitter(), dtype=default_float())
                Kmn = kernel(Z, offset_x)
                Knn = kernel(offset_y, full=False)

                mu0, _ = base_conditional(Kmn, Kmm, Knn, q_mu,
                                            full_cov=False, white=self.whiten, q_sqrt=None)

                n, _ = base_conditional(Kmn, Kmm, Knn, tf.ones_like(q_mu),
                                        full_cov=False, white=self.whiten, q_sqrt=None)
                q_mus.append( q_mu - ((mu0 - offset_y) / n) )

            return tf.concat(q_mus, axis=0)



    def predict_fs(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=True) -> tf.Tensor:

        # build conditional statistics
        Z = self.inducing_variables
        Kmm = self.K1(Z).to_dense() + eye(self.num_inducing, value=default_jitter(), dtype=Xnew.dtype)
        Kmn = self.K2(Z, Xnew)
        Knn = self.K3(Xnew).to_dense()

        q_mu = self.q_mu
        q_sqrt = self.q_sqrt
        Kmn = tf.transpose(Kmn, (1, 2, 0))
        mean, var = base_conditional(Kmn, Kmm, Knn, q_mu, full_cov=True,
                                         q_sqrt=q_sqrt,
                                         white=self.whiten)
        mean, var = mean[..., 0], var[:, 0]

        mean += self.offsets_y

        if full_output_cov:
            return mean, var
        else:
            return mean, tf.linalg.diag_part(var)

    def sample_fs(self, Xnew, num_samples=10):

        mean, var = self.predict_fs(Xnew)
        L = tf.linalg.cholesky(var)
        eps = tf.random.normal([num_samples]+mean.shape+[1], dtype=default_float())
        return (L @ eps + mean[..., None])

    def predict_means(self, Xnew: tf.Tensor) -> tf.Tensor:
        mus = []
        for c in range(self.num_components):
            mu, var = conditional(Xnew,
                              self.inducing_variables[c],
                              self.kernels[c],
                              self.q_mus[c],
                              q_sqrt=None,
                              full_cov=False,
                              white=self.whiten,
                              full_output_cov=False)

            mus.append(mu + self.mean_functions[c](Xnew))
        return tf.concat(mus, axis=-1)


class SparseVariationalGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

    def __init__(self,
                 kernel,
                 likelihood,
                 inducing_variable,
                 *,
                 mean_function=None,
                 num_latent: int = 1,
                 q_diag: bool = False,
                 q_mu=None,
                 q_sqrt=None,
                 whiten: bool = True,
                 num_data=None,
                 offset=None):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # init the super class, accept args
        super().__init__(kernel, likelihood, mean_function, num_latent)
        self.num_data = num_data
        self.q_diag = q_diag
        self.whiten = whiten

        # init variational parameters
        self.q = SparseGP(self.kernel, inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
                          offset=offset)

    def prior_kl(self):
        return kullback_leiblers.prior_kl(self.q.inducing_variable,
                                          self.kernel,
                                          self.q.q_mu,
                                          self.q.q_sqrt,
                                          whiten=self.whiten)

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()
        f_mean, f_var = self.q.predict_f(X, full_cov=False, full_output_cov=False)
        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def elbo(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.log_marginal_likelihood(data)

    def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:

        return self.q.predict_f(Xnew, full_cov=full_cov, full_output_cov=full_output_cov)


class SparseVariationalMeanFieldGPs(BayesianModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

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
                 offsets_x=None,
                 offsets_y=None,
                 deterministic_optimisation=False,
                 num_samples=10
                 ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """

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

        # init variational parameters
        self.deterministic_optimisation = deterministic_optimisation
        self.num_samples = num_samples

        self.q = MeanFieldSparseGPs(kernels,
                                    inducing_variables,
                                    mean_functions=mean_functions,
                                    q_diag=q_diag,
                                    q_mus=q_mus,
                                    q_sqrts=q_sqrts,
                                    whiten=whiten,
                                    offsets_x=offsets_x,
                                    offsets_y=offsets_y
                                    )

    def prior_kls(self):
        kls = []
        for q in self.q.qs:
            kls.append(
                kullback_leiblers.prior_kl(q.inducing_variable,
                                          q.kernel,
                                          q.q_mu,
                                          q.q_sqrt,
                                          whiten=q.whiten)
            )
        return kls

    def prior_kl(self):
        return tf.reduce_sum(self.prior_kls())

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()
        f_means, f_vars = self.q.predict_fs(X, full_cov=False, full_output_cov=False)

        # hard code additive
        f_mean = tf.reduce_sum(f_means, axis=-1, keepdims=True)
        f_var = tf.reduce_sum(f_vars, axis=-1, keepdims=True)

        var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def elbo(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.log_marginal_likelihood(data)

    def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:
        f_means, f_vars = self.q.predict_fs(Xnew, full_cov=full_cov,
                                            full_output_cov=full_output_cov)
        # hard coded additive
        f_mean = tf.reduce_sum(f_means, axis=-1, keepdims=True)
        f_var = tf.reduce_sum(f_vars, axis=-1, keepdims=True)
        return f_mean, f_var


class SparseVariationalCoupledGPs(BayesianModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::
      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """

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
                 offsets_x=None,
                 offsets_y=None,
                 deterministic_optimisation=False,
                 num_samples=10
                 ):
        """
        - kernel, likelihood, inducing_variables, mean_function are appropriate
          GPflow objects
        - num_latent is the number of latent processes to use, defaults to 1
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        - num_data is the total number of observations, defaults to X.shape[0]
          (relevant when feeding in external minibatches)
        """

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

        self.deterministic_optimisation = deterministic_optimisation
        self.num_samples = num_samples

        self.q = SparseCoupledGPs(kernels,
                 inducing_variables,
                 mean_functions=mean_functions,
                 q_diag=q_diag,
                 q_mus=q_mus,
                 q_sqrts=q_sqrts,
                 offsets_x=offsets_x,
                 offsets_y=offsets_y,
                 whiten=whiten)

    def prior_kl(self):
        if self.whiten:
            return kullback_leiblers.gauss_kl(self.q.q_mu, self.q.q_sqrt, None)
        else:
            Kuu = self.q.K1(self.q.inducing_variables).to_dense()  + \
                  ops.eye(self.q.num_inducing, value=default_jitter(), dtype=default_float()) # [P, M, M] or [M, M]
            return kullback_leiblers.gauss_kl(self.q.q_mu, self.q.q_sqrt, Kuu)

    def log_likelihood(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This gives a variational bound on the model likelihood.
        """
        X, Y = data
        kl = self.prior_kl()

        if self.deterministic_optimisation:
            raise NotImplementedError()
            f_means, f_vars = self.q.predict_fs(X, full_cov=False, full_output_cov=True)

            # hard code additive
            f_mean = tf.reduce_prod(f_means, axis=-1, keepdims=True)
            f_var = tf.reduce_prod(f_vars, axis=[-1, -2])[..., None]

            var_exp = self.likelihood.variational_expectations(f_mean, f_var, Y)
        else:
            fs_samples = self.q.sample_fs(X)
            pred = self.predictor(fs_samples)  # tf.reduce_prod(fs_samples, axis=-2)
            var_exp = self.likelihood.log_prob(pred, Y)

        if self.num_data is not None:
            num_data = tf.cast(self.num_data, kl.dtype)
            minibatch_size = tf.cast(tf.shape(X)[0], kl.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, kl.dtype)
        return tf.reduce_sum(var_exp) * scale - kl

    def elbo(self, data: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        """
        This returns the evidence lower bound (ELBO) of the log marginal likelihood.
        """
        return self.log_marginal_likelihood(data)

    def predictor(self, F):
        return F[..., 0, :] * F[..., 1, :] + F[..., 2, :]

    # def predict_f(self, Xnew: tf.Tensor, full_cov=False, full_output_cov=False) -> tf.Tensor:
    #     f_means, f_vars = self.qs.predict_fs(Xnew, full_cov=full_cov,
    #                                          full_output_cov=full_output_cov)
    #     # hard coded additive
    #     f_mean = tf.reduce_sum(f_means, axis=-1, keepdims=True)
    #     f_var = tf.reduce_sum(f_vars, axis=-1, keepdims=True)
    #     return f_mean, f_var
