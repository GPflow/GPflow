import numpy as np
import tensorflow as tf

from gpflow import settings,  densities, transforms, kullback_leiblers, features, conditionals
from gpflow.core.compilable import Build
from gpflow.params import Parameterized, ParamList, DataHolder, Parameter, Minibatch, DataHolder
from gpflow.decors import autoflow,params_as_tensors, params_as_tensors_for
from gpflow.mean_functions import Zero
from gpflow.kernels import Kernel
from gpflow.conditionals import conditional
from gpflow.features import InducingPoints
from gpflow.kullback_leiblers import gauss_kl
float_type = settings.float_type

def reparameterize(mean, var, z, full_cov=False):
    """
    Implements the 'reparameterization trick' for the Gaussian, either full rank or diagonal
    If the z is a sample from N(0, 1), the output is a sample from N(mean, var)
    If full_cov=True then var must be of shape S,N,N,D and the full covariance is used. Otherwise
    var must be S,N,D and the operation is elementwise
    :param mean: mean of shape S,N,D
    :param var: covariance of shape S,N,D or S,N,N,D
    :param z: samples form unit Gaussian of shape S,N,D
    :param full_cov: bool to indicate whether var is of shape S,N,N,D or S,N,D
    :return sample from N(mean, var) of shape S,N,D
    """
    if full_cov is False:
        return mean + z * (var + settings.jitter) ** 0.5

    else:
        S, N, D = tf.shape(mean)[0], tf.shape(mean)[1], tf.shape(mean)[2] # var is SNND
        mean = tf.transpose(mean, (0, 2, 1))  # SND -> SDN
        var = tf.transpose(var, (0, 3, 1, 2))  # SNND -> SDNN
        I = settings.jitter * tf.eye(N, dtype=settings.float_type)[None, None, :, :] # 11NN
        chol = tf.cholesky(var + I)  # SDNN
        z_SDN1 = tf.transpose(z, [0, 2, 1])[:, :, :, None]  # SND->SDN1
        f = mean + tf.matmul(chol, z_SDN1)[:, :, :, 0]  # SDN(1)
        return tf.transpose(f, (0, 2, 1)) # SND

class Latent(Parameterized):
    """If g is a latent with inducing u_g then this represents 
    p(g,u_g)
    """
    def __init__(self, Z, mean_function, kern, num_latent=1, whiten=True, name=None):
        super(Latent, self).__init__(name=name)
        self.mean_function = mean_function
        self.kern = kern
        self.num_latent = num_latent
        M = Z.shape[0]
        
        self.feature = InducingPoints(Z)
        num_inducing = len(self.feature)
        self.whiten = whiten
        
        self.q_mu = Parameter(np.zeros((num_inducing, self.num_latent), dtype=settings.float_type))
        
        q_sqrt = np.tile(np.eye(M)[None, :, :], [self.num_latent, 1, 1])
        transform = transforms.LowerTriangular(M, num_matrices=self.num_latent)
        self.q_sqrt = Parameter(q_sqrt, transform=transform)
    
    @autoflow((settings.float_type, [None, None]))
    def predict_f(self, Xnew):
        """
        Compute the mean and variance of the latent function(s) at the points
        Xnew.
        """
        return self.conditional(Xnew[None,:,:])

    @autoflow((settings.float_type, [None, None]))
    def predict_f_full_cov(self, Xnew):
        """
        Compute the mean and covariance matrix of the latent function(s) at the
        points Xnew.
        """
        return self.conditional(Xnew[None,:,:], full_cov=True)
            
    def sample_from_conditional(self, X, z=None, full_cov=False):
        """
        Calculates self.conditional and also draws a sample
        If z=None then the tensorflow random_normal function is used to generate the
        N(0, 1) samples, otherwise z are used for the whitened sample points
        :param X: Input locations (S,N,D_in)
        :param full_cov: Whether to compute correlations between outputs
        :param z: None, or the sampled points in whitened representation
        :return: samples (S,N,D), mean (S,N,D), var (S,N,N,D or S,N,D)
        """
        mean, var = self.conditional(X, full_cov=full_cov)
        if z is None:
            z = tf.random_normal(tf.shape(mean), dtype=settings.float_type)
        samples = reparameterize(mean, var, z, full_cov=full_cov)
        return samples, mean, var
    
    @params_as_tensors
    def conditional(self, X, full_cov=False):
        """
        If g is a latent with inducing u_g and this represents 
        p(g,u_g) then this computes the mean and covariance of
        p(g|u_g)
    
        A multisample conditional, where X is shape (S,N,D_out), independent over samples S
        if full_cov is True
            mean is (S,N,D_out), var is (S,N,N,D_out)
        if full_cov is False
            mean and var are both (S,N,D_out)
        :param X:  The input locations (S,N,D_in)
        :param full_cov: Whether to calculate full covariance or just diagonal
        :return: mean (S,N,D_out), var (S,N,D_out or S,N,N,D_out)
        """
        def single_sample_conditional(X, full_cov=False):
            mean, var = conditional(X, self.feature.Z, self.kern,
                        self.q_mu, q_sqrt=self.q_sqrt,
                        full_cov=full_cov, white=self.whiten)
            return mean + self.mean_function(X), var

        if full_cov is True:
            f = lambda a: single_sample_conditional(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = single_sample_conditional(X_flat)
            return [tf.reshape(m, [S, N, -1]) for m in [mean, var]]
        
    def KL(self):
        """
        The KL divergence from the variational distribution to the prior
        :return: KL divergence from N(q_mu, q_sqrt) to N(0, I), independently for each GP
        """
        return gauss_kl(self.q_mu, self.q_sqrt)
  
