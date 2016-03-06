from __future__ import absolute_import
import numpy as np
import tensorflow as tf
from .model import GPModel
from .param import Param
from .conditionals import gp_predict_whitened
from .priors import Gaussian
from .mean_functions import Zero


class SGPMC(GPModel):
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=Zero(), num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        Z is a data matrix, of inducing inputs, size M x D
        kern, likelihood, mean_function are appropriate GPflow objects

        This is the Sparse Variational GP using MCMC (SGPMC). The key reference is

        @inproceedings{hensman2015mcmc,
          title={MCMC for Variatinoally Sparse Gaussian Processes},
          author={Hensman, James and Matthews, Alexander G. de G. and Filippone, Maurizio and Ghahramani, Zoubin},
          booktitle={Proceedings of NIPS},
          year={2015}
        }

        The latent function values are represented by centered
        (whitened) variables, so

            v ~ N(0, I)
            u = Lv

        with

            L L^T = K
        
        """
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.num_data = X.shape[0]
        self.num_inducing = Z.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.Z = Z # Z is not a parameter!
        self.V = Param(np.zeros((self.num_inducing, self.num_latent)))
        self.V.prior = Gaussian(0., 1.)

    def build_likelihood(self):
        """
        This function computes the (log) optimal distribution for v, q*(v). 
        """
        #get the (marginals of) q(f): exactly predicting!
        fmean, fvar = gp_predict_whitened(self.X, self.Z, self.kern, self.V)
        fmean += self.mean_function(self.X)
        return tf.reduce_sum( self.likelihood.variational_expectations(fmean, fvar, self.Y) )

    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | (U=LV) )

        where F* are points on the GP at Xnew, F=LV are points on the GP at Z, 

        """
        mean, var = gp_predict_whitened(Xnew, self.Z, self.kern, self.V)
        return mean + self.mean_function(Xnew), var

