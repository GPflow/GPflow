import tensorflow as tf
import numpy as np
from .model import GPModel
from .gpr import GPR
from .param import Param, AutoFlow, DataHolder
from .mean_functions import Zero
from . import likelihoods
from .tf_wraps import eye
from . import transforms
from . import kernels
from ._settings import settings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

float_type = settings.dtypes.float_type


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
        likelihood for the training data.
        """
        return self._build_likelihood_graph(self.X_mean, self.X_var, self.Y, self.X_prior_mean, self.X_prior_var)

    def _build_likelihood_graph(self, X_mean, X_var, Y, X_prior_mean=None, X_prior_var=None):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood given a Gaussian multivariate distribution representing
        X (and its priors) and observed Y
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

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]), (float_type, [None, None]))
    def held_out_data_objective(self, Y_new, mu_new, var_new):
        """
        TF computation of likelihood objective + gradients, given new observed points and a candidate q(X*)
        :param Y_new: new observed points, size Nnew (number of new points) x D (dimensions).
        :param mu_new: candidate mean, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :param var_new: candidate variance, np.ndarray of size Nnew (number of new points) x Q (latent dimensions)
        :return: returning an (objective,gradients) tuple. gradients is a list of 2 matrices for mu and var of size
        Nnew x Q
        """
        X_mean = tf.concat(0, [self.X_mean, mu_new])
        X_var = tf.concat(0, [self.X_var, var_new])
        Y = tf.concat(0, [self.Y, Y_new])
        objective = self._build_likelihood_graph(X_mean, X_var, Y)

        # Collect gradients
        gradients = tf.gradients(objective, [mu_new, var_new])

        f = tf.negative(objective, name='objective')
        g = tf.negative(gradients, name='grad_objective')
        return f, g

    def _held_out_data_wrapper_creator(self, Y_new):
        """
        Private wrapper function for returning an objective function accepted by scipy.optimize.minimize
        :param Y_new: new observed points, size Nnew (number of new points) x D (dimensions).
        :return: function accepting a flat numpy array of size 2 x Nnew (number of new points) x Q (latent dimensions)
        and returning an (objective,gradient) tuple
        """
        infer_number = Y_new.shape[0]
        num_param = infer_number * self.num_latent * 2

        def fun(x_flat):
            # Unpack q(X*) candidate
            mu_new = x_flat[:num_param/2].reshape((infer_number, self.num_latent))
            var_new = x_flat[num_param/2:].reshape((infer_number, self.num_latent))

            # Compute likelihood & flatten gradients
            f,g = self.held_out_data_objective(Y_new, mu_new, var_new)
            return f, np.hstack(map(lambda gradient: gradient.flatten(), g))

        return fun

    def infer_latent_inputs(self, Y_new, method='L-BFGS-B', tol=None, return_logprobs=False, **kwargs):
        """
        Computes the latent representation of new observed points by maximizing
        .. math::

            p(Y*|Y)

        :param Y_new: new observed points, size Nnew (number of new points) x D (dimensions).
        :param method: method is a string (default 'L-BFGS-B') specifying the scipy optimization routine
        :param tol: tol is the tolerance to be passed to the optimization routine
        :param kern: kernel specification, by default RBF
        :param return_logprobs: returns the likelihood probability after optimization
        :param kwargs: passed on to the options field of the scipy minimizer

        :returns (mean, var) or (mean, var, prob) in case return_logprobs is true.
        :rtype mean, var: np.ndarray, size Nnew (number of new points ) x Q (latent dim)
        """

        infer_number = Y_new.shape[0]
        assert (Y_new.shape[1] == self.output_dim)

        # Objective
        f = self._held_out_data_wrapper_creator(Y_new)

        # Initialization: could do this with tf?
        nearest_idx = np.argmin(cdist(self.Y.value, Y_new), axis=0)
        x_init = np.hstack((self.X_mean.value[nearest_idx, :].flatten(),
                            self.X_var.value[nearest_idx, :].flatten()))

        # Optimize - restrict var to be positive
        result = minimize(fun=f,
                          x0=x_init,
                          jac=True,
                          method=method,
                          tol=tol,
                          bounds = [(None, None)]*(x_init.size/2) + [(0, None)]*(x_init.size/2),
                          options=kwargs)
        x_hat = result.x
        mu = x_hat[:infer_number * self.num_latent].reshape((infer_number, self.num_latent))
        var = x_hat[infer_number * self.num_latent:].reshape((infer_number, self.num_latent))

        if return_logprobs:
            return mu, var, -result.fun
        else:
            return mu, var

    @AutoFlow((float_type, [None, None]), (float_type, [None, None]))
    def uncertain_predict(self, Xstarmu, Xstarvar):
        num_inducing = tf.shape(self.Z)[0]
        num_predict = tf.shape(Xstarmu)[0]
        num_out = tf.shape(self.Y)[1]

        # Kernel expectations, w.r.t q(X) and q(X*)
        psi1 = self.kern.eKxz(self.Z, self.X_mean, self.X_var) # num_train x num_inducing
        psi2 = tf.reduce_sum(self.kern.eKzxKxz(self.Z, self.X_mean, self.X_var), 0) # num_inducing x num_inducing
        psi0star = self.kern.eKdiag(Xstarmu, Xstarvar) # num_predict
        psi1star = self.kern.eKxz(self.Z, Xstarmu, Xstarvar) # num_predict x num_inducing
        psi2star = tf.reduce_sum(self.kern.eKzxKxz(self.Z, Xstarmu, Xstarvar), 0) # num_inducing x num_inducing

        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6 # num_inducing x num_inducing
        sigma2 = self.likelihood.variance
        sigma = tf.sqrt(sigma2)
        L = tf.cholesky(Kuu) # num_inducing x num_inducing

        A = tf.matrix_triangular_solve(L, tf.transpose(psi1), lower=True) / sigma # num_inducing x num_train
        tmp = tf.matrix_triangular_solve(L, psi2, lower=True) # num_inducing x num_inducing
        AAT = tf.matrix_triangular_solve(L, tf.transpose(tmp), lower=True) / sigma2
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B) # num_inducing x num_inducing
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma # num_inducing x num_out
        tmp1 = tf.matrix_triangular_solve(L, tf.transpose(psi1star), lower=True)
        tmp2 = tf.matrix_triangular_solve(LB, tmp1, lower=True)
        mean = tf.matmul(tf.transpose(tmp2), c)

        L3 = tf.tile(tf.expand_dims(L, 0), [num_predict, 1, 1]) # num_predict x num_inducing x num_inducing
        LB3 = tf.tile(tf.expand_dims(LB, 0), [num_predict, 1, 1]) # num_predict x num_inducing x num_inducing
        tmp3 = tf.matrix_triangular_solve(LB3, tf.matrix_triangular_solve(L3, tf.einsum('ij,ik->ijk', psi1star, psi1star))) # num_predict x num_inducing x num_inducing
        tmp4 = tf.matrix_triangular_solve(LB3, tf.matrix_triangular_solve(L3, tf.transpose(tmp3, perm=[0, 2, 1]))) # num_predict x num_inducing x num_inducing
        tmp5 = tf.matrix_triangular_solve(L, tf.transpose(tf.matrix_triangular_solve(L, psi2star))) # num_inducing x num_inducing
        c3 = tf.tile(tf.expand_dims(c, 0), [num_predict, 1, 1]) # num_predict x num_inducing x num_out

        # some segments - must be joined into sigma
        TT = tf.trace(tmp5 - tf.matrix_triangular_solve(LB, tf.transpose(tf.matrix_triangular_solve(LB, psi2star)))) # num_predict
        var1 = tf.matmul(tf.transpose(c3, perm=[0,2,1]), tf.matmul(tmp4,c3)) # num_predict x num_out x num_out
        var2 = tf.einsum('i,jk->ijk', psi0star - TT, tf.eye(num_out)) # num_predict x num_out x num_out
        var = var1 + var2
        return mean, var


    def predict_unobserved_f(self, Ynew_observed, observed):
        observed = np.atleast_1d(observed)
        assert(Ynew_observed.shape[1] == observed.size)

        # Retain observed variables of training data, obtain q(X*)
        Y_full = self.Y
        self.Y = DataHolder(Y_full.value[:, observed])
        self.output_dim = observed.size
        Xstarmu, Xstarvar = self.infer_latent_inputs(Ynew_observed)

        # Perform prediction w.r.t q(X*)


        # Create full predictive distribution for Ynew
        #unobserved = np.setdiff1d(np.arange(self.Y.shape[1]), observed)
        #mu = np.zeros((Ynew_observed.shape[0], self.output_dim))
        #mu[:, observed] = Ynew_observed
        #mu[:, unobserved] = unobserved_mu
        #var = np.zeros((Ynew_observed.shape[0], self.output_dim))
        #return mu, var


