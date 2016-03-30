import tensorflow as tf
import numpy as np
from param import Param
from .model import GPModel
import transforms
import conditionals
from .mean_functions import Zero
from tf_hacks import eye
import kullback_leiblers


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    @inproceedings{hensman2014scalable,
      title={Scalable Variational Gaussian Process Classification},
      author={Hensman, James and Matthews, Alexander G. de G. and Ghahramani, Zoubin},
      booktitle={Proceedings of AISTATS},
      year={2015}
    }

    """
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=Zero(), num_latent=None, q_diag=False, whiten=True, minibatch=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, likelihood, mean_function are appropriate GPflow objects
        Z is a matrix of pseudo inputs, size M x D
        num_latent is the number of latent process to use, default to Y.shape[1]
        q_diag is a boolean. If True, the covariance is approximated by a diagonal matrix.
        whiten is a boolean. It True, we use the whitened represenation of the inducing points.
        """
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]

        #these variables used for minibatching
        self._tfX = tf.Variable(self.X, name="tfX")
        self._tfY = tf.Variable(self.Y, name="tfY")
        self.minibatch = minibatch


        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)), transforms.positive)
        else:
            self.q_sqrt = Param(np.array([np.eye(self.num_inducing) for _ in range(self.num_latent)]).swapaxes(0,2))

    @property
    def minibatch(self):
        if self._minibatch is None:
            return len(self.X)
        else:
            return self._minibatch

    @minibatch.setter
    def minibatch(self, val):
        self._minibatch = val

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt, self.num_latent)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt, self.num_latent)
        else:
            K = self.kern.Kzz(self.Z) + eye(self.num_inducing) * 1e-6
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K, self.num_latent)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K, self.num_latent)
        return KL

    def _compile(self, optimizer=None):
        """
        compile the tensorflow function "self._objective"
        """
        opt_step = GPModel._compile(self, optimizer)
        def obj(x):
            if self.minibatch / float(len(self.X)) > 0.5:
                ss = np.random.permutation(len(self.X))[:self.minibatch]
            else:
                # This is much faster than above, and for N >> minibatch, it doesn't make much difference. This actually
                # becomes the limit when N is around 10**6, which isn't uncommon when using SVI.
                ss = np.random.randint(len(self.X), size=self.minibatch)
            return self._session.run([self._minusF, self._minusG], feed_dict={self._free_vars: x,
                                                                              self._tfX: self.X[ss, :],
                                                                              self._tfY: self.Y[ss, :]})
        self._objective = obj
        return opt_step

    def optimize(self, method='L-BFGS-B', callback=None, max_iters=1000, **kw):
        def calc_feed_dict():
            ss = np.random.randint(len(self.dX), size=self.minibatch)
            return {self.X: self.dX[ss, :], self.Y: self.dY[ss, :]}

        return GPModel.optimize(self, method, callback, max_iters, calc_feed_dict, **kw)

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        #Get prior KL.
        KL  = self.build_prior_KL()
    
        #Get conditionals
        if self.whiten:
            fmean, fvar = conditionals.gaussian_gp_predict_whitened(self._tfX, self.Z, self.kern, self.q_mu, self.q_sqrt, self.num_latent)
        else:
            fmean, fvar = conditionals.gaussian_gp_predict(self._tfX, self.Z, self.kern, self.q_mu, self.q_sqrt, self.num_latent)

        #add in mean function to conditionals.
        fmean += self.mean_function(self._tfX)
        
        #Get variational expectations.
        variational_expectations = self.likelihood.variational_expectations(fmean, fvar, self._tfY)

        minibatch_scale = len(self.X) / tf.cast(tf.shape(self._tfX)[0], tf.float64)
        return tf.reduce_sum(variational_expectations) * minibatch_scale - KL

    def build_predict(self, Xnew):
        if self.whiten:
            mu, var =  conditionals.gaussian_gp_predict_whitened(Xnew, self.Z, self.kern, self.q_mu, self.q_sqrt, self.num_latent)
        else:
            mu, var =  conditionals.gaussian_gp_predict(Xnew, self.Z, self.kern, self.q_mu, self.q_sqrt, self.num_latent)
        return mu + self.mean_function(Xnew), var


      


