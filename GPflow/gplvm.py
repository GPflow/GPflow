import tensorflow as tf
import numpy as np
from .model import GPModel
from .gpr import GPR
from .param import Param, AutoFlow, DataHolder
from .mean_functions import Constant, Zero
from . import likelihoods
from .tf_wraps import eye
from . import transforms
from . import kernels
from ._settings import settings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist
from contextlib import contextmanager

float_type = settings.dtypes.float_type
int_type = settings.dtypes.int_type



def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.
    :param X: data array of size N (number of points) x D (dimensions)
    :param Q: Number of latent dimensions, Q < D
    :return: PCA projection array of size N x Q.
    """
    assert Q <= X.shape[1], 'Cannot have more latent dimensions than observed'
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X - X.mean(0)).dot(W)


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the latent X.
    """

    def __init__(self, Y, latent_dim, X_mean=None, kern=None, mean_function=Zero()):
        """
        Initialise GPLVM object. This method only works with a Gaussian likelihood.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions)
        :param X_mean: latent positions (N x Q), for the initialisation of the latent space.
        :param kern: kernel specification, by default RBF
        :param mean_function: mean function, by default None.
        """
        if kern is None:
            kern = kernels.RBF(latent_dim, ARD=True)
        if X_mean is None:
            X_mean = PCA_reduce(Y, latent_dim)
        assert X_mean.shape[1] == latent_dim, \
            'Passed in number of latent ' + str(latent_dim) + ' does not match initial X ' + str(X_mean.shape[1])
        self.num_latent = X_mean.shape[1]
        assert Y.shape[1] >= self.num_latent, 'More latent dimensions than observed.'
        GPR.__init__(self, X_mean, Y, kern, mean_function=mean_function)
        del self.X  # in GPLVM this is a Param
        self.X = Param(X_mean)


class BayesianGPLVM(GPModel):
    def __init__(self, X_mean, X_var, Y, kern, M, Z=None, X_prior_mean=None, X_prior_var=None):
        """
        Initialise Bayesian GPLVM object. This method only works with a Gaussian likelihood.
        :param X_mean: initial latent positions, size N (number of points) x Q (latent dimensions).
        :param X_var: variance of latent positions (N x Q), for the initialisation of the latent space.
        :param Y: data matrix, size N (number of points) x D (dimensions)
        :param kern: kernel specification, by default RBF
        :param M: number of inducing points
        :param Z: matrix of inducing points, size M (inducing points) x Q (latent dimensions). By default
        random permutation of X_mean.
        :param X_prior_mean: prior mean used in KL term of bound. By default 0. Same size as X_mean.
        :param X_prior_var: pripor variance used in KL term of bound. By default 1.
        """
        GPModel.__init__(self, X_mean, Y, kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X  # in GPLVM this is a Param
        self.X_mean = Param(X_mean)
        # diag_transform = transforms.DiagMatrix(X_var.shape[1])
        # self.X_var = Param(diag_transform.forward(transforms.positive.backward(X_var)) if X_var.ndim == 2 else X_var,
        #                    diag_transform)
        assert X_var.ndim == 2
        self.X_var = Param(X_var, transforms.positive)
        self.num_data = X_mean.shape[0]
        self.output_dim = Y.shape[1]

        assert np.all(X_mean.shape == X_var.shape)
        assert X_mean.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        assert X_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

        # inducing points
        if Z is None:
            # By default we initialize by subset of initial latent points
            Z = np.random.permutation(X_mean.copy())[:M]
        else:
            assert Z.shape[0] == M
        self.Z = Param(Z)
        self.num_latent = Z.shape[1]
        assert X_mean.shape[1] == self.num_latent

        # deal with parameters for the prior mean variance of X
        if X_prior_mean is None:
            X_prior_mean = np.zeros((self.num_data, self.num_latent))
        self.X_prior_mean = X_prior_mean
        if X_prior_var is None:
            X_prior_var = np.ones((self.num_data, self.num_latent))
        self.X_prior_var = X_prior_var

        assert X_prior_mean.shape[0] == self.num_data
        assert X_prior_mean.shape[1] == self.num_latent
        assert X_prior_var.shape[0] == self.num_data
        assert X_prior_var.shape[1] == self.num_latent

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood for the training data (all dimensions).
        """

        # Default: construct likelihood graph using the training data Y and initialized q(X)
        return self._build_likelihood_graph(self.X_mean, self.X_var, self.Y, self.X_prior_mean, self.X_prior_var)

    def _build_likelihood_graph(self, X_mean, X_var, Y, X_prior_mean=None, X_prior_var=None):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood given a Gaussian multivariate distribution representing
        X (and its priors) and observed Y

        Split from the general build_likelihood method, as the graph is reused by the held_out_data_objective
        method for inference of latent points for new data points
        """

        if X_prior_mean is None:
            X_prior_mean = tf.zeros((tf.shape(Y)[0], self.num_latent), float_type)
        if X_prior_var is None:
            X_prior_var = tf.ones((tf.shape(Y)[0], self.num_latent), float_type)

        num_inducing = tf.shape(self.Z)[0]
        psi0 = tf.reduce_sum(self.kern.eKdiag(X_mean, X_var), 0)
        psi1 = self.kern.eKxz(self.Z, X_mean, X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, X_mean, X_var), 0)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        log_det_B = 2. * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        dX_var = X_var if len(X_var.get_shape()) == 2 else tf.matrix_diag_part(X_var)
        NQ = tf.cast(tf.size(X_mean), float_type)
        D = tf.cast(tf.shape(Y)[1], float_type)
        KL = -0.5 * tf.reduce_sum(tf.log(dX_var)) \
             + 0.5 * tf.reduce_sum(tf.log(X_prior_var)) \
             - 0.5 * NQ \
             + 0.5 * tf.reduce_sum((tf.square(X_mean - X_prior_mean) + dX_var) / X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(Y), float_type)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.diag_part(AAT)))
        bound -= KL
        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points.
        Note that this is very similar to the SGPR prediction, for which
        there are notes in the SGPR notebook.
        :param Xnew: Point to predict at.
        """
        num_inducing = tf.shape(self.Z)[0]
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var)
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        Kus = self.kern.K(self.Z, Xnew)
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu)

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True)
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma
        tmp1 = tf.matrix_triangular_solve(L, Kus, lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)
        if full_cov:
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2) \
                  - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0) \
                  - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.stack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var

    def build_predict_distribution(self, Xstarmu, Xstarvar):
        """
        Build the graph to map a latent point to its corresponding output. Unlike build_predict, here a
        Gaussian density is mapped rather than only its mean as in build_predict. Details of the calculation
        are in the gplvm predict x notebook.

        :param Xstarmu: mean of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :param Xstarvar: variance of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :return: tensor for computation of the moments of the output distribution.
        """
        num_inducing = tf.shape(self.Z)[0] # M
        num_predict = tf.shape(Xstarmu)[0] # N*
        num_out = self.output_dim     # p

        # Kernel expectations, w.r.t q(X) and q(X*)
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var) # N x M
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0) # M x M
        psi0star = self.kern.eKdiag(Xstarmu, Xstarvar) # N*
        psi1star = self.kern.eKxz(self.Z, Xstarmu, Xstarvar) # N* x M
        psi2star = self.kern.eKzxKxz(self.Z, Xstarmu, Xstarvar) # N* x M x M

        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6 # M x M
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu) # M x M

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma # M x N
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True) # M x M
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B) # M x M
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma # M x p
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)

        # All of these: N* x M x M
        L3 = tf.tile(tf.expand_dims(L, 0), [num_predict, 1, 1])
        LB3 = tf.tile(tf.expand_dims(LB, 0), [num_predict, 1, 1])
        tmp3 = tf.matrix_triangular_solve(LB3, tf.matrix_triangular_solve(L3, tf.expand_dims(psi1star, -1)))
        tmp4 = tf.matmul(tmp3, tf.transpose(tmp3, perm=[0, 2, 1]))
        tmp5 = tf.matrix_triangular_solve(L3, tf.transpose(tf.matrix_triangular_solve(L3, psi2star), perm=[0, 2, 1]))
        tmp6 = tf.matrix_triangular_solve(LB3, tf.transpose(tf.matrix_triangular_solve(LB3, tmp5), perm=[0, 2, 1]))

        c3 = tf.tile(tf.expand_dims(c, 0), [num_predict, 1, 1])  # N* x M x p
        TT = tf.trace(tmp5 - tmp6)  # N*
        diagonals = tf.einsum("ij,k->ijk", eye(num_out), psi0star - TT) # p x p x N*
        covar1 = tf.matmul(tf.transpose(c3, perm=[0, 2, 1]), tf.matmul(tmp6 - tmp4, c3))  # N* x p x p
        covar2 = tf.transpose(diagonals, perm=[2, 0, 1])  # N* x p x p
        covar = covar1 + covar2
        return mean + self.mean_function(Xstarmu), covar

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]),
              (float_type, [None, None]), (int_type, [None]))
    def held_out_data_objective(self, Ynew, mu, var, observed):
        """
        TF computation of likelihood objective + gradients, given new observed points and a candidate q(X*)
        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions), with k <= D.
        :param mu: candidate mean, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :param var: candidate variance, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :param observed: indices for the observed dimensions np.ndarray of size k
        :return: returning a tuple (objective,gradients). gradients is a list of 2 matrices for mu and var of size
        Nnew x Q
        """
        idx = tf.expand_dims(observed, -1)
        Y_obs = tf.transpose(tf.gather_nd(tf.transpose(self.Y), idx))
        X_mean = tf.concat([self.X_mean, mu], 0)
        X_var = tf.concat([self.X_var, var], 0)
        Y = tf.concat([Y_obs, Ynew], 0)

        # Build the likelihood graph for the suggested q(X,X*) and the observed dimensions of Y and Y*
        objective = self._build_likelihood_graph(X_mean, X_var, Y)

        # Collect gradients
        gradients = tf.gradients(objective, [mu, var])

        f = tf.negative(objective, name='objective')
        g = tf.negative(gradients, name='grad_objective')
        return f, g

    def _held_out_data_wrapper_creator(self, Ynew, observed):
        """
        Private wrapper function for returning an objective function accepted by scipy.optimize.minimize
        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions)
        :return: function accepting a flat numpy array of size 2 * Nnew (number of new points) * Q (latent dimensions)
        and returning a tuple (objective,gradient)
        """
        infer_number = Ynew.shape[0]
        half_num_param = infer_number * self.num_latent

        def fun(x_flat):
            # Unpack q(X*) candidate
            mu_new = x_flat[:half_num_param].reshape((infer_number, self.num_latent))
            var_new = x_flat[half_num_param:].reshape((infer_number, self.num_latent))

            # Compute likelihood & flatten gradients
            f,g = self.held_out_data_objective(Ynew, mu_new, var_new, observed)
            return f, np.hstack(map(lambda gradient: gradient.flatten(), g))

        return fun

    def infer_latent_inputs(self, Ynew, method='L-BFGS-B', tol=None, return_logprobs=False, observed=None, **kwargs):
        """
        Computes the latent representation of new observed points by maximizing
        .. math::

            p(Y*|Y)

        It is automatically assumed all dimensions D were observed unless the observed parameter is specified.

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions). with k <= D.
        :param method: method is a string (default 'L-BFGS-B') specifying the scipy optimization routine
        :param tol: tol is the tolerance to be passed to the optimization routine
        :param kern: kernel specification, by default RBF
        :param return_logprobs: return the likelihood probability after optimization (default: False)
        :param observed: list of dimensions specified with length k. None (the default) indicates all D were observed
        :param kwargs: passed on to the options field of the scipy minimizer

        :returns (mean, var) or (mean, var, prob) in case return_logprobs is true.
        :rtype mean, var: np.ndarray, size Nnew (number of new points ) x Q (latent dim)
        """

        observed = np.arange(self.Y.shape[1], dtype=np.int32) if observed is None else np.atleast_1d(observed)
        assert (Ynew.shape[1] == observed.size)
        infer_number = Ynew.shape[0]

        # Initialization: could do this with tf?
        nearest_idx = np.argmin(cdist(self.Y.value[:, observed], Ynew), axis=0)
        x_init = np.hstack((self.X_mean.value[nearest_idx, :].flatten(),
                            self.X_var.value[nearest_idx, :].flatten()))

        # Objective
        f = self._held_out_data_wrapper_creator(Ynew, observed)

        # Optimize - restrict var to be positive
        result = minimize(fun=f,
                          x0=x_init,
                          jac=True,
                          method=method,
                          tol=tol,
                          bounds = [(None, None)]*int(x_init.size/2) + [(0, None)]*int(x_init.size/2),
                          options=kwargs)
        x_hat = result.x
        mu = x_hat[:infer_number * self.num_latent].reshape((infer_number, self.num_latent))
        var = x_hat[infer_number * self.num_latent:].reshape((infer_number, self.num_latent))

        if return_logprobs:
            return mu, var, -result.fun
        else:
            return mu, var

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_f_uncertain(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the (non-Gaussian) distribution of the latent function by propagating a
        Gaussian distribution.

        Note: this method is only available in combination with Constant or Zero mean functions.

        :param Xstarmu: mean of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :param Xstarvar: variance of the points in latent space size: Nnew (number of new points ) x Q (latent dim)
        :returns (mean, covar)
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D
        covar: np.ndarray, size Nnew (number of new points ) x D x D
        """
        assert(isinstance(self.mean_function, (Zero, Constant)))
        return self.build_predict_distribution(Xstarmu, Xstarvar)

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def predict_y_uncertain(self, Xstarmu, Xstarvar):
        """
        Predicts the first and second moment of the (non-Gaussian) distribution by propagating a
        Gaussian distribution.

        Note: this method is only available in combination with Constant or Zero mean functions.
        """
        assert(isinstance(self.mean_function, (Zero, Constant)))
        mean, covar = self.build_predict_distribution(Xstarmu, Xstarvar)
        num_predict = tf.shape(mean)[0]
        num_out = tf.shape(mean)[1]
        noise = tf.tile(tf.expand_dims(self.likelihood.variance * eye(num_out), 0), [num_predict, 1, 1])
        return mean, covar+noise

    def predict_f_unobserved(self, Ynew, observed):
        """
        Given a partial observation, predict the first and second moments of the non-Gaussian distriubtion over the
        unobserved part of the latent functions:
        .. math::

            p(F^U_* | Y^O_*, X, X_*)

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions),
        with k <= D.
        :param observed: 1D list or array of indices of observed dimensions, size D-k. Should at least contain one
        observed dimension, and at most all dimensions.
        :returns (mean, covar) of non-Gaussian predictive distribution over the unobserved dimensions
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D-k
        covar: np.ndarray, size Nnew (number of new points ) x D-k x D-k
        """
        observed = np.unique(np.atleast_1d(observed))
        assert(0 < observed.size <= self.output_dim)
        unobserved = np.setdiff1d(np.arange(self.output_dim), observed)
        assert(Ynew.shape[1] == observed.size)

        # obtain q(X*), only consider observed dimensions
        Xstarmu, Xstarvar = self.infer_latent_inputs(Ynew, observed=observed)

        # Perform (full) prediction w.r.t q(X*)
        unobserved_mu, unobserved_covar = self.predict_f_uncertain(Xstarmu, Xstarvar)

        # Return only predictions for unobserved dimensions
        return unobserved_mu[:, unobserved], \
               unobserved_covar[:, unobserved, :][:, :, unobserved]

    def predict_y_unobserved(self, Ynew, observed):
        """
        Given a partial observation, predict the first and second moments of the non-Gaussian distriubtion over the
        unobserved part:
        .. math::

            p(Y^U_* | Y^O_*, X, X_*)

        :param Ynew: new observed points, size Nnew (number of new points) x k (observed dimensions),
        with k <= D.
        :param observed: 1D list or array of indices of observed dimensions, size D-k. Should at least contain one
        observed dimension, and at most all dimensions.
        :returns (mean, covar) of non-Gaussian predictive distribution over the unobserved dimensions
        :rtype mean: np.ndarray, size Nnew (number of new points ) x D-k
        covar: np.ndarray, size Nnew (number of new points ) x D-k x D-k
        """
        observed = np.unique(np.atleast_1d(observed))
        assert(0 < observed.size <= self.output_dim)
        unobserved = np.setdiff1d(np.arange(self.output_dim), observed)
        assert(Ynew.shape[1] == observed.size)

        # obtain q(X*), only consider observed dimensions
        Xstarmu, Xstarvar = self.infer_latent_inputs(Ynew, observed=observed)

        # Perform (full) prediction w.r.t q(X*)
        unobserved_mu, unobserved_covar = self.predict_y_uncertain(Xstarmu, Xstarvar)

        # Return only predictions for unobserved dimensions
        return unobserved_mu[:, unobserved], \
               unobserved_covar[:, unobserved, :][:, :, unobserved]
