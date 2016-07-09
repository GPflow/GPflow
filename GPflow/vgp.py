import tensorflow as tf
import numpy as np
from .param import Param, DataHolder
from .model import GPModel
from . import transforms
from .mean_functions import Zero
from .tf_hacks import eye


class VGP(GPModel):
    def __init__(self, X, Y, kern, likelihood,
                 mean_function=Zero(), num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects

        This is the variational objective for the Variational Gaussian Process
        (VGP). The key reference is:

        @article{Opper:2009,
            title = {The Variational Gaussian Approximation Revisited},
            author = {Opper, Manfred and Archambeau, Cedric},
            journal = {Neural Comput.},
            year = {2009},
            pages = {786--792},
        }

        The idea is that the posterior over the function-value vector F is
        approximated by a Gaussian, and the KL divergence is minimised between
        the approximation and the posterior. It turns out that the optimal
        posterior precision shares off-diagonal elements with the prior, so
        only the diagonal elements of the precision need be adjusted.

        The posterior approximation is

        q(f) = N(f | K alpha, [K^-1 + diag(square(lambda))]^-1)

        """
        X = DataHolder(X, on_shape_change='recompile')
        Y = DataHolder(Y, on_shape_change='recompile')
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
        self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                              transforms.positive)

    def _compile(self, optimizer=None):
        """
        Before calling the standard compile function, check to see if the size
        of the data has changed and add variational parameters appropriately.

        This is necessary because the hape of the parameters depends on the
        shape of the data.
        """
        if not self.num_data == self.X.shape[0]:
            self.num_data = self.X.shape[0]
            self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
            self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                                  transforms.positive)
        super(VGP, self)._compile(optimizer=optimizer)

    def build_likelihood(self):
        """
        q_alpha, q_lambda are variational parameters, size N x R

        This method computes the variational lower bound on the likelihood,
        which is:

            E_{q(F)} [ \log p(Y|F) ] - KL[ q(F) || p(F)]

        with

            q(f) = N(f | K alpha + mean, [K^-1 + diag(square(lambda))]^-1) .

        """
        K = self.kern.K(self.X)
        K_alpha = tf.matmul(K, self.q_alpha)
        f_mean = K_alpha + self.mean_function(self.X)
        # for each of the data-dimensions (columns of Y), find the diagonal
        # of the variance, and also relevant parts of the KL.
        f_var = []
        A_logdet = tf.zeros((1,), tf.float64)
        trAi = tf.zeros((1,), tf.float64)
        for d in range(self.num_latent):
            b = self.q_lambda[:, d]
            B = tf.expand_dims(b, 1)
            A = eye(self.num_data) + K*B*tf.transpose(B)
            L = tf.cholesky(A)
            Li = tf.matrix_triangular_solve(L, eye(self.num_data), lower=True)
            LiBi = Li / b

            # full_sigma:return tf.diag(b**-2) - LiBi.T.dot(LiBi)
            f_var.append(1./tf.square(b) - tf.reduce_sum(tf.square(LiBi), 0))
            A_logdet += 2*tf.reduce_sum(tf.log(tf.diag_part(L)))
            trAi += tf.reduce_sum(tf.square(Li))

        f_var = tf.transpose(tf.pack(f_var))

        KL = 0.5 * (A_logdet + trAi - self.num_data * self.num_latent
                    + tf.reduce_sum(K_alpha*self.q_alpha))

        v_exp = self.likelihood.variational_expectations(f_mean, f_var, self.Y)
        return tf.reduce_sum(v_exp) - KL

    def build_predict(self, Xnew, full_cov=False):
        """
        The posterior variance of F is given by

            q(f) = N(f | K alpha + mean, [K^-1 + diag(lambda**2)]^-1)

        Here we project this to F*, the values of the GP at Xnew which is given
        by

           q(F*) = N ( F* | K_{*F} alpha + mean, K_{**} - K_{*f}[K_{ff} +
                                           diag(lambda**-2)]^-1 K_{f*} )

        """

        # compute kernel things
        Kx = self.kern.K(Xnew, self.X)
        K = self.kern.K(self.X)

        # predictive mean
        f_mean = tf.matmul(Kx, self.q_alpha) + self.mean_function(Xnew)

        # predictive var
        f_var = []
        for d in range(self.num_latent):
            b = self.q_lambda[:, d]
            A = K + tf.diag(1./tf.square(b))
            L = tf.cholesky(A)
            LiKx = tf.matrix_triangular_solve(L, tf.transpose(Kx), lower=True)
            if full_cov:
                f_var.append(self.kern.K(Xnew) -
                             tf.matmul(tf.transpose(LiKx), LiKx))
            else:
                f_var.append(self.kern.Kdiag(Xnew) -
                             tf.reduce_sum(tf.square(LiKx), 0))
        f_var = tf.pack(f_var)
        return f_mean, tf.transpose(f_var)
