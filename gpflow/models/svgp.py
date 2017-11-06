# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, Mark van der Wilk
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
import tensorflow as tf
import numpy as np

from .. import settings
from .. import transforms
from .. import conditionals
from .. import kullback_leiblers

from ..params import Parameter
from ..params import Minibatch
from ..params import DataHolder

from ..decors import params_as_tensors
from ..decors import autoflow

from ..models.model import GPModel


class SVGP(GPModel):
    """
    This is the Sparse Variational GP (SVGP). The key reference is

    ::

      @inproceedings{hensman2014scalable,
        title={Scalable Variational Gaussian Process Classification},
        author={Hensman, James and Matthews,
                Alexander G. de G. and Ghahramani, Zoubin},
        booktitle={Proceedings of AISTATS},
        year={2015}
      }

    """
    def __init__(self, X, Y, kern, likelihood, Z,
                 mean_function=None,
                 num_latent=None,
                 q_diag=False,
                 whiten=True,
                 minibatch_size=None,
                 num_data=None,
                 **kwargs):
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
        - minibatch_size, if not None, turns on mini-batching with that size.
        - num_data is the total number of observations, default to X.shape[0]
          (relevant when feeding in external minibatches)
        """
        # sort out the X, Y into MiniBatch objects if required.
        if minibatch_size is None:
            X = DataHolder(X)
            Y = DataHolder(Y)
        else:
            X = Minibatch(X, batch_size=minibatch_size, seed=0)
            Y = Minibatch(Y, batch_size=minibatch_size, seed=1)

        # init the super class, accept args
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_data = num_data or X.shape[0]
        self.q_diag, self.whiten = q_diag, whiten
        self.Z = Parameter(Z)
        self.num_latent = num_latent or Y.shape[1]
        self.num_inducing = Z.shape[0]

        # init variational parameters
        self.q_mu = Parameter(np.zeros((self.num_inducing, self.num_latent), dtype=settings.np_float))
        if self.q_diag:
            self.q_sqrt = Parameter(np.ones((self.num_inducing, self.num_latent), dtype=settings.np_float),
                                transforms.positive)
        else:
            q_sqrt = np.array([np.eye(self.num_inducing, dtype=settings.np_float)
                               for _ in range(self.num_latent)]).swapaxes(0, 2)
            self.q_sqrt = Parameter(q_sqrt, transform=transforms.LowerTriangular(self.num_inducing, self.num_latent))


    @params_as_tensors
    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.kern.K(self.Z) + tf.eye(self.num_inducing, dtype=settings.tf_float) * settings.numerics.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

    @params_as_tensors
    def _build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self._build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.tf_float) / tf.cast(tf.shape(self.X)[0], settings.tf_float)

        return tf.reduce_sum(var_exp) * scale - KL

    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var

    @autoflow()
    @params_as_tensors
    def linear_weights_posterior(self):
        """
        Some kernels have finite dimensional feature maps. Others although not having finite
        feature maps can have approximated feature vectors see eg.

        ::
            @inproceedings{rahimi2008random,
              title={Random features for large-scale kernel machines},
              author={Rahimi, Ali and Recht, Benjamin},
              booktitle={Advances in neural information processing systems},
              pages={1177--1184},
              year={2008}
            }

        With these features, GP regression can be seen as Bayesian linear regression with Gaussian
        priors on the initial weights vector. See Section 2.1 of:
        ::
            @book{rasmussen2006gaussian,
              title={Gaussian processes for machine learning},
              author={Rasmussen, Carl Edward and Williams, Christopher KI},
              volume={1},
              year={2006},
              publisher={MIT press Cambridge}
            }


        This method compute the posterior mean and the lower trainglular decomposition of the
        precision matrix for the distribution over the
        linear weights.
        Note that this method may not always work. If the kernel does not have a feature mapping
        (even a random approximation) then a NotImplementedError will be raised.
        :returns mean, matrix of precision/variance, flag set to true is matrix is variance.
        """
        assert self.num_latent == 1, "Only yet implemented for one latent variable GP."
        # We squeeze the q_sqrt below to get it for one latent factor, this needs to be changed
        # if want more latent values in addition to perhaps other stuff.

        feats = self.kern._feature_map(self.Z)
        num_obs = tf.shape(feats)[0]
        num_feats = tf.shape(feats)[1]

        kernel_at_z_true = self.kern.K(self.Z)
        chol_kzz_true = tf.cholesky(kernel_at_z_true + tf.eye(num_obs, dtype=settings.tf_float)
                                    * settings.numerics.jitter_level)

        kernel_at_z_approx = tf.matmul(feats, feats, transpose_b=True)
        chol_kzz_approx = tf.cholesky(kernel_at_z_approx + tf.eye(num_obs, dtype=settings.tf_float)
                                      * settings.numerics.jitter_level)

        # === Mean ===
        if self.whiten:
            # empirically we have found that going from the whitened representation to the
            # non-whitened representation first using the true kernel and then using the kernel
            # approximation to compute the posterior mean of the weights works better.
            # however may still give poor estimates away from inducing points.
            u = tf.matmul(chol_kzz_true, self.q_mu)
        else:
            u = self.q_mu

        # Going via the O(N^3) complexity route:
        Kzzinv_u = tf.cholesky_solve(chol_kzz_approx, u)
        mean = tf.matmul(feats, Kzzinv_u, transpose_a=True)

        # === Variance ===
        LiPhi = tf.matrix_triangular_solve(chol_kzz_approx, feats)

        if self.whiten:
            R = tf.matmul(chol_kzz_true, tf.squeeze(self.q_sqrt))
        else:
            R = tf.squeeze(self.q_sqrt)
        LitLiPhi = tf.matrix_triangular_solve(tf.transpose(chol_kzz_approx), LiPhi, lower=False)

        QsrttLiPhi = tf.matmul(R, LitLiPhi, transpose_a=True)

        term1 = tf.matmul(QsrttLiPhi, QsrttLiPhi, transpose_a=True)

        # term 2
        term2 = -tf.matmul(LiPhi, LiPhi, transpose_a=True)

        variance = term1 + term2 + tf.eye(num_feats, num_feats, dtype=settings.tf_float)

        return mean, variance, tf.constant(True, dtype=tf.bool)





