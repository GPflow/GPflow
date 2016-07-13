import tensorflow as tf
import numpy as np
from .model import GPModel
from .param import Param, DataHolder
from .mean_functions import Zero
from . import likelihoods
from .tf_hacks import eye


class SGPR(GPModel):

    def __init__(self, X, Y, kern, Z, mean_function=Zero()):
        """
        This Sparse Variational GP regression. The key reference is

        @inproceedings{titsias2009variational,
          title={Variational learning of inducing variables in
                 sparse Gaussian processes},
          author={Titsias, Michalis K},
          booktitle={International Conference on
                     Artificial Intelligence and Statistics},
          pages={567--574},
          year={2009}
        }


        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood. For a derivation of the terms in here, see the associated
        SGPR notebook.
        """

        num_inducing = tf.shape(self.Z)[0]
        num_data = tf.cast(tf.shape(self.Y)[0], tf.float64)
        output_dim = tf.cast(tf.shape(self.Y)[1], tf.float64)

        err = self.Y - self.mean_function(self.X)
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        L = tf.cholesky(Kuu)
        sigma = tf.sqrt(self.likelihood.variance)

        # Compute intermediate matrices
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        AAT = tf.matmul(A, tf.transpose(A))
        B = AAT + eye(num_inducing)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma

        # compute log marginal bound
        bound = -0.5 * num_data * output_dim * np.log(2 * np.pi)
        bound += - output_dim * tf.reduce_sum(tf.log(tf.diag_part(LB)))
        bound -= 0.5 * num_data * output_dim * tf.log(self.likelihood.variance)
        bound += -0.5*tf.reduce_sum(tf.square(err))/self.likelihood.variance
        bound += 0.5*tf.reduce_sum(tf.square(c))
        bound += -0.5 * tf.reduce_sum(Kdiag) / self.likelihood.variance
        bound += 0.5 * tf.reduce_sum(tf.diag_part(AAT))

        return bound

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew. For a derivation of the terms in here, see the associated SGPR
        notebook.
        """
        num_inducing = tf.shape(self.Z)[0]
        err = self.Y - self.mean_function(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6
        Kus = self.kern.K(self.Z, Xnew)
        sigma = tf.sqrt(self.likelihood.variance)
        L = tf.cholesky(Kuu)
        A = tf.matrix_triangular_solve(L, Kuf, lower=True) / sigma
        B = tf.matmul(A, tf.transpose(A)) + eye(num_inducing)
        LB = tf.cholesky(B)
        Aerr = tf.matmul(A, err)
        c = tf.matrix_triangular_solve(LB, Aerr, lower=True) / sigma
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


class GPRFITC(GPModel):

    def __init__(self, X, Y, kern, Z, mean_function=Zero()):
        """
        This implements GP regression with the FITC approximation.
        The key reference is

        @inproceedings{Snelson06sparsegaussian,
        author = {Edward Snelson and Zoubin Ghahramani},
        title = {Sparse Gaussian Processes using Pseudo-inputs},
        booktitle = {Advances In Neural Information Processing Systems },
        year = {2006},
        pages = {1257--1264},
        publisher = {MIT press}
        }

        Implementation loosely based on code from GPML matlab library although
        obviously gradients are automatic in GPflow.

        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a matrix of pseudo inputs, size M x D
        kern, mean_function are appropriate GPflow objects

        This method only works with a Gaussian likelihood.

        """
        X = DataHolder(X, on_shape_change='pass')
        Y = DataHolder(Y, on_shape_change='pass')
        likelihood = likelihoods.Gaussian()
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.Z = Param(Z)
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]

    def build_common_terms(self):
        num_inducing = tf.shape(self.Z)[0]
        err = self.Y - self.mean_function(self.X)  # size N x R
        Kdiag = self.kern.Kdiag(self.X)
        Kuf = self.kern.K(self.Z, self.X)
        Kuu = self.kern.K(self.Z) + eye(num_inducing) * 1e-6

        Luu = tf.cholesky(Kuu)  # => Luu^T Luu = Kuu
        V = tf.matrix_triangular_solve(Luu, Kuf)  # => V^T V = Qff

        diagQff = tf.reduce_sum(tf.square(V), 0)
        nu = Kdiag - diagQff + self.likelihood.variance

        B = eye(num_inducing) + tf.matmul(V / nu, tf.transpose(V))
        L = tf.cholesky(B)
        beta = err / tf.expand_dims(nu, 1)  # size N x R
        alpha = tf.matmul(V, beta)  # size N x R

        gamma = tf.matrix_triangular_solve(L, alpha, lower=True)  # size N x R

        return err, nu, Luu, L, alpha, beta, gamma

    def build_likelihood(self):
        """
        Construct a tensorflow function to compute the bound on the marginal
        likelihood.
        """

        # FITC approximation to the log marginal likelihood is
        # log ( normal( y | mean, K_fitc ) )
        # where K_fitc = Qff + diag( \nu )
        # where Qff = Kfu Kuu^{-1} Kuf
        # with \nu_i = Kff_{i,i} - Qff_{i,i} + \sigma^2

        # We need to compute the Mahalanobis term -0.5* err^T K_fitc^{-1} err
        # (summed over functions).

        # We need to deal with the matrix inverse term.
        # K_fitc^{-1} = ( Qff + \diag( \nu ) )^{-1}
        #            = ( V^T V + \diag( \nu ) )^{-1}
        # Applying the Woodbury identity we obtain
        #            = \diag( \nu^{-1} ) - \diag( \nu^{-1} ) V^T ( I + V \diag( \nu^{-1} ) V^T )^{-1) V \diag( \nu^{-1} )
        # Let \beta =  \diag( \nu^{-1} ) err
        # and let \alpha = V \beta
        # then Mahalanobis term = -0.5* ( \beta^T err - \alpha^T Solve( I + V \diag( \nu^{-1} ) V^T, alpha ) )

        err, nu, Luu, L, alpha, beta, gamma = self.build_common_terms()

        mahalanobisTerm = -0.5 * tf.reduce_sum(tf.square(err) / tf.expand_dims(nu, 1)) + 0.5 * tf.reduce_sum(tf.square(gamma))

        # We need to compute the log normalizing term -N/2 \log 2 pi - 0.5 \log \det( K_fitc )

        # We need to deal with the log determinant term.
        # \log \det( K_fitc ) = \log \det( Qff + \diag( \nu ) )
        #                    = \log \det( V^T V + \diag( \nu ) )
        # Applying the determinant lemma we obtain
        #                    = \log [ \det \diag( \nu ) \det( I + V \diag( \nu^{-1} ) V^T ) ]
        #                    = \log [ \det \diag( \nu ) ] + \log [ \det( I + V \diag( \nu^{-1} ) V^T ) ]

        constantTerm = -0.5 * self.num_data * tf.log(tf.constant(2. * np.pi, tf.float64))
        logDeterminantTerm = -0.5 * tf.reduce_sum(tf.log(nu)) - tf.reduce_sum(tf.log(tf.diag_part(L)))
        logNormalizingTerm = constantTerm + logDeterminantTerm

        return mahalanobisTerm + logNormalizingTerm

    def build_predict(self, Xnew, full_cov=False):
        """
        Compute the mean and variance of the latent function at some new points
        Xnew.
        """
        _, _, Luu, L, _, _, gamma = self.build_common_terms()
        Kus = self.kern.K(self.Z, Xnew)  # size  M x Xnew

        w = tf.matrix_triangular_solve(Luu, Kus, lower=True)  # size M x Xnew

        tmp = tf.matrix_triangular_solve(tf.transpose(L), gamma, lower=False)
        mean = tf.matmul(tf.transpose(w), tmp) + self.mean_function(Xnew)
        intermediateA = tf.matrix_triangular_solve(L, w, lower=True)

        if full_cov:
            var = self.kern.K(Xnew) - tf.matmul(tf.transpose(w), w) + tf.matmul(tf.transpose(intermediateA), intermediateA)
            var = tf.tile(tf.expand_dims(var, 2), tf.pack([1, 1, tf.shape(self.Y)[1]]))
        else:
            var = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(w), 0) + tf.reduce_sum(tf.square(intermediateA), 0)  # size Xnew,
            var = tf.tile(tf.expand_dims(var, 1), tf.pack([1, tf.shape(self.Y)[1]]))

        return mean, var
