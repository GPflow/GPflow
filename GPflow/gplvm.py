import tensorflow as tf
import numpy as np
from .model import GPModel
from .gpr import GPR
from .param import Param
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye
from . import kernel_expectations as ke
from . import transforms


def PCA_reduce(X, Q):
    """
    A helpful function for linearly reducing the dimensionality of the data X
    to Q.

    X is a N x D numpy array.
    Q is an integer, Q < D

    returns a numpy array, N x Q
    """
    evecs, evals = np.linalg.eigh(np.cov(X.T))
    i = np.argsort(evecs)[::-1]
    W = evals[:, i]
    W = W[:, :Q]
    return (X-X.mean(0)).dot(W)


class GPLVM(GPR):
    """
    Standard GPLVM where the likelihood can be optimised with respect to the  latent X.
    """
    def __init__(self, Y, X_mean, kern, mean_function=Zero()):
        """
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        X_mean is a matrix, size N x Q, for the initialisation of the latent space.
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        self.num_latent = X_mean.shape[1]
        assert Y.shape[1] >= self.num_latent, 'More latent dimensions than observed.'
        GPR. __init__(self, X_mean, Y, kern, mean_function=mean_function)
        del self.X  # in GPLVM this is a Param
        self.X = Param(X_mean)


class BayesianGPLVM(GPModel):

    def __init__(self, X_mean, X_var, Y, kern, Z, X_prior_mean=None, X_prior_var=None):
        """
        X_mean is a data matrix, size N x D
        X_var is a data matrix, size N x D (X_var > 0)
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        GPModel.__init__(self, X_mean, Y, kern, likelihood=likelihoods.Gaussian(), mean_function=Zero())
        del self.X
        self.X_mean = Param(X_mean)
        self.X_var = Param(X_var, transforms.positive)
        self.Z = Param(Z)
        self.num_data = X_mean.shape[0]
        self.num_latent = Z.shape[1]
        self.output_dim = Y.shape[1]

        assert np.all(X_mean.shape == X_var.shape)
        assert X_mean.shape[1] == self.num_latent
        assert X_mean.shape[0] == Y.shape[0], 'X mean and Y must be same size.'
        assert X_var.shape[0] == Y.shape[0], 'X var and Y must be same size.'

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
        likelihood.
        """
        num_inducing = tf.shape(self.Z)[0]

        psi0, psi1, psi2 = ke.build_psi_stats(self.Z, self.kern, self.X_mean, self.X_var)
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
        c = tf.matrix_triangular_solve(LB, tf.matmul(A, self.Y), lower=True) / sigma

        # KL[q(x) || p(x)]
        NQ = tf.cast(tf.size(self.X_mean), tf.float64)
        D = tf.cast(tf.shape(self.Y)[1], tf.float64)
        KL = -0.5*tf.reduce_sum(tf.log(self.X_var)) \
            + 0.5*tf.reduce_sum(tf.log(self.X_prior_var))\
            - 0.5 * NQ\
            + 0.5 * tf.reduce_sum((tf.square(self.X_mean - self.X_prior_mean) + self.X_var) / self.X_prior_var)

        # compute log marginal bound
        ND = tf.cast(tf.size(self.Y), tf.float64)
        bound = -0.5 * ND * tf.log(2 * np.pi * sigma2)
        bound += -0.5 * D * log_det_B
        bound += -0.5 * tf.reduce_sum(tf.square(self.Y)) / sigma2
        bound += 0.5 * tf.reduce_sum(tf.square(c))
        bound += -0.5 * D * (tf.reduce_sum(psi0) / sigma2 -
                             tf.reduce_sum(tf.diag_part(AAT)))
        bound -= KL

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. Note that this is very similar to the SGPR prediction, for whcih
        there are notes in the SGPR notebook.
        """
        num_inducing = tf.shape(self.Z)[0]
        psi0, psi1, psi2 = ke.build_psi_stats(self.Z, self.kern, self.X_mean, self.X_var)
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
            var = self.kern.K(Xnew) + tf.matmul(tf.transpose(tmp2), tmp2)\
                - tf.matmul(tf.transpose(tmp1), tmp1)
            shape = tf.pack([1, 1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 2), shape)
        else:
            var = self.kern.Kdiag(Xnew) + tf.reduce_sum(tf.square(tmp2), 0)\
                - tf.reduce_sum(tf.square(tmp1), 0)
            shape = tf.pack([1, tf.shape(self.Y)[1]])
            var = tf.tile(tf.expand_dims(var, 1), shape)
        return mean + self.mean_function(Xnew), var
