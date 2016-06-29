import tensorflow as tf
import numpy as np
from .param import Param, DataHolder
from .model import GPModel
from . import transforms
from . import conditionals
from .mean_functions import Zero
from .tf_hacks import eye
from . import kullback_leiblers


class MinibatchData(DataHolder):
    """
    A special DataHolder class which feeds a minibatch to tensorflow via
    get_feed_dict().
    """
    def __init__(self, array, minibatch_size, rng=None):
        """
        array is a numpy array of data.
        minibatch_size (int) is the size of the minibatch
        rng is an instance of np.random.RandomState(), defaults to seed 0.
        """
        DataHolder.__init__(self, array, on_shape_change='pass')
        self.minibatch_size = minibatch_size
        self.rng = rng or np.random.RandomState(0)

    def generate_index(self):
        if float(self.minibatch_size) / float(self._array.shape[0]) > 0.5:
            return self.rng.permutation(self._array.shape[0])[:self.minibatch_size]
        else:
            # This is much faster than above, and for N >> minibatch,
            # it doesn't make much difference. This actually
            # becomes the limit when N is around 10**6, which isn't
            # uncommon when using SVI.
            return self.rng.randint(self._array.shape[0], size=self.minibatch_size)

    def get_feed_dict(self):
        return {self._tf_array: self._array[self.generate_index()]}


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    @inproceedings{hensman2014scalable,
      title={Scalable Variational Gaussian Process Classification},
      author={Hensman, James and Matthews,
              Alexander G. de G. and Ghahramani, Zoubin},
      booktitle={Proceedings of AISTATS},
      year={2015}
    }

    """
    def __init__(self, X, Y, kern, likelihood, Z, mean_function=Zero(),
                 num_latent=None, q_diag=False, whiten=True, minibatch_size=None):
        """
        - X is a data matrix, size N x D
        - Y is a data matrix, size N x R
        - kern, likelihood, mean_function are appropriate GPflow objects
        - Z is a matrix of pseudo inputs, size M x D
        - num_latent is the number of latent process to use, default to
          Y.shape[1]
        - q_diag is a boolean. If True, the covariance is approximated by a
          diagonal matrix.
        - whiten is a boolean. If True, we use the whitened representation of
          the inducing points.
        """
        # sort out the X, Y into MiniBatch objects.
        if minibatch_size is None:
            minibatch_size = X.shape[0]
        self.num_data = X.shape[0]
        X = MinibatchData(X, minibatch_size, np.random.RandomState(0))
        Y = MinibatchData(Y, minibatch_size, np.random.RandomState(0))

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Param(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]

        # init variational parameters
        self.q_mu = Param(np.zeros((self.num_inducing, self.num_latent)))
        if self.q_diag:
            self.q_sqrt = Param(np.ones((self.num_inducing, self.num_latent)),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Param(q_sqrt)

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu,
                                                           self.q_sqrt,
                                                           self.num_latent)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu,
                                                      self.q_sqrt,
                                                      self.num_latent)
        else:
            K = self.kern.K(self.Z) + eye(self.num_inducing) * 1e-6
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu,
                                                     self.q_sqrt,
                                                     K,
                                                     self.num_latent)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu,
                                                self.q_sqrt,
                                                K,
                                                self.num_latent)
        return KL

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        if self.whiten:
            cond_fn = conditionals.gaussian_gp_predict_whitened
        else:
            cond_fn = conditionals.gaussian_gp_predict
        fmean, fvar = cond_fn(self.X, self.Z, self.kern,
                              self.q_mu, self.q_sqrt, self.num_latent)

        # add in mean function to conditionals.
        fmean += self.mean_function(self.X)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, tf.float64) / tf.cast(tf.shape(self.X)[0], tf.float64)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew, full_cov=False):
        if self.whiten:
            cond_fn = conditionals.gaussian_gp_predict_whitened
        else:
            cond_fn = conditionals.gaussian_gp_predict
        mu, var = cond_fn(Xnew, self.Z, self.kern,
                          self.q_mu, self.q_sqrt, self.num_latent, full_cov)
        return mu + self.mean_function(Xnew), var
