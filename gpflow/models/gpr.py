# Copyright 2016 James Hensman, Valentine Svensson, alexggmatthews, fujiisoup
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

from gpflow import likelihoods

from gpflow import settings
from gpflow.models.model import GPModel
from gpflow.densities import multivariate_normal
from gpflow.params import DataHolder
from gpflow.decors import params_as_tensors
from gpflow.decors import name_scope
from gpflow.decors import autoflow


class GPR(GPModel):
    """
    Gaussian Process Regression.

    This is a vanilla implementation of GP regression with a Gaussian
    likelihood.  Multiple columns of Y are treated independently.

    The log likelihood i this models is sometimes referred to as the 'marginal log likelihood', and is given by

    .. math::

       \\log p(\\mathbf y \\,|\\, \\mathbf f) = \\mathcal N\\left(\\mathbf y\,|\, 0, \\mathbf K + \\sigma_n \\mathbf I\\right)
    """
    def __init__(self, X, Y, kern, mean_function=None, random_seed_for_random_features=897, **kwargs):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_latent = Y.shape[1]
        self.random_seed_for_random_features = random_seed_for_random_features

    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        """
        Construct a tensorflow function to compute the likelihood.

            \log p(Y | theta).

        """
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.tf_float) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict

        This method computes

            p(F* | Y )

        where F* are points on the GP at Xnew, Y are noisy observations at X.

        """
        Kx = self.kern.K(self.X, Xnew)
        K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.tf_float) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Y - self.mean_function(self.X))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Y)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Y)[1]])
        return fmean, fvar

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
        :returns mean, matrix of precision/variance, flag set to true is matrix is variance opr false for precision
        """
        assert self.num_latent == 1, "Only yet implemented for one latent variable GP."
        # Pretty sure that it should work fine for more dimensional latent GP but just need
        # to check that TF's Cholesky solve can deal with this.

        # This follows almost exactly the example 2.1 of GPML as we have a tractable likelihood
        feats = self.kern._feature_map(self.X)
        num_obs = tf.shape(feats)[0]
        num_feats = tf.shape(feats)[1]
        # NB we currently run a naive version. However, if number of data points is smaller than
        # the feature dimension then I think we can use the Matrix Inversion Lemma to cut down on
        # computation
        Sigma_obs_inversed = tf.eye(num_obs, num_obs, dtype=settings.tf_float) / self.likelihood.variance

        A = tf.matmul(feats, tf.matmul(Sigma_obs_inversed, feats), transpose_a=True) + \
            tf.eye(num_feats, num_feats, dtype=settings.tf_float)
        L_A = tf.cholesky(A)

        unscaled_mean = tf.matmul(feats, tf.matmul(Sigma_obs_inversed, (self.Y - self.mean_function(self.X))),
                                  transpose_a=True)
        mean = tf.cholesky_solve(L_A, unscaled_mean)

        # we return the precision matrix Cholesky decomposed as this is useful representation when
        # wanting to sample from this function.
        precision_mat = A
        return mean, precision_mat, tf.constant(False, dtype=tf.bool)

