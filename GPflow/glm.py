import numpy as np
from .model import GPModel
from .param import Param
import tensorflow as tf

class GLM(GPModel):
    """ 
    A generalized Linear model
    """
    def __init__(self, X, Y, likelihood):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood are appropriate GPt objects

        we construct a set on linear weights, size D x R

        """
        GPModel.__init__(self, X, Y, None, likelihood) # hack:kern is none
        self.num_data = X.shape[0]
        self.num_latent = Y.shape[1]
        self.input_dim = X.shape[1]
        self.W = Param(np.zeros((self.input_dim, self.num_latent)))
        self.b = Param(np.zeros(self.num_latent))


    def build_likelihood(self):
        """
        Constuct a tensorflow function to compute the likelihood of this GLM

            \log p(Y | W, b).

        """
        F = tf.matmul(self.X, self.W) + self.b
        return tf.reduce_sum(self.likelihood.logp(F, self.Y))

    def build_predict(self, Xnew):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | W, b)

        where F* are predictions of the model at Xnew

        """
        mean =  tf.matmul(Xnew, self.W) + self.b
        return mean, tf.zeros_like(mean)

