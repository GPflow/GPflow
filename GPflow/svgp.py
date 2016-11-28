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
from .param import Param, DataHolder
from .model import GPModel
from . import transforms, conditionals, kullback_leiblers
from .mean_functions import Zero
from .tf_wraps import eye
from ._settings import settings

class IndexManager(object):
    """
    Base clase for methods of batch indexing data.
    rng is an instance of np.random.RandomState, defaults to seed 0.
    """
    def __init__(self, minibatch_size, total_points, rng = None):
        self.minibatch_size = minibatch_size
        self.total_points = total_points
        self.rng = rng or np.random.RandomState(0)
        
    def nextIndeces(self):
        raise NotImplementedError

class ReplacementSampling(IndexManager):
	def nextIndeces(self):
		return self.rng.randint(self.total_points, 
		                        size=self.minibatch_size)

class NoReplacementSampling(IndexManager):
	def nextIndeces(self):
		permutation = self.rng.permutation(self.total_points)
		return permutation[:self.minibatch_size]

class SequenceIndeces(IndexManager):
    """
    A class that maintains the state necessary to manage 
    sequential indexing of data holders.
    """
    def __init__(self, minibatch_size, total_points, rng = None):
        self.counter = 0
        IndexManager.__init__(self, minibatch_size, total_points, rng)
		
    def nextIndeces(self):
        """
		Written so that if total_points
		changes this will still work
        """
        firstIndex = self.counter
        lastIndex = self.counter + self.minibatch_size
        self.counter = lastIndex % self.total_points
        return np.arange(firstIndex,lastIndex) % self.total_points

class MinibatchData(DataHolder):
    """
    A special DataHolder class which feeds a minibatch 
    to tensorflow via update_feed_dict().
    """
    
    #List of valid specifiers for generation methods.
    _generation_methods = ['replace','noreplace','sequential']
    
    def __init__(self, array, 
                       minibatch_size, 
                       rng=None, 
                       batch_manager=None):
        """
        array is a numpy array of data.
        minibatch_size (int) is the size of the minibatch
        
        batch_manager specified data sampling scheme and is a subclass 
        of IndexManager.
        
        Note: you may want to randomize the order of the data 
        if using sequential generation.
        """
        DataHolder.__init__(self, array, on_shape_change='pass')
        total_points = self._array.shape[0]
        self.parseGenerationMethod(batch_manager, 
                                   minibatch_size, 
                                   total_points,
                                   rng)
        
    def parseGenerationMethod(self, 
                              input_batch_manager, 
                              minibatch_size,
                              total_points,
                              rng):
        #Logic for default behaviour.
        #When minibatch_size is a small fraction of total_point
        #ReplacementSampling should give similar results to 
        #NoReplacementSampling and the former can be much faster.
        if input_batch_manager==None: 
		    fraction = float(minibatch_size) / float(total_points)
		    if fraction > 0.5:
		        self.index_manager = ReplacementSampling(minibatch_size,
		                                                 total_points,
		                                                 rng)
		    else:
		        self.index_manager = NoReplacementSampling(minibatch_size,
                                                           total_points,
                                                           rng)
        else: #Explicitly specified behaviour.
		    if input_batch_manager.__class__ not in IndexManager.__subclasses__():
			    raise NotImplementedError
		    self.index_manager = input_batch_manager
			
    def update_feed_dict(self, key_dict, feed_dict):
        next_indeces = self.index_manager.nextIndeces()
        feed_dict[key_dict[self]] = self._array[next_indeces]


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
            self.q_sqrt = Param(q_sqrt)  # , transforms.LowerTriangular(q_sqrt.shape[2]))  # Temp remove transform

    def build_prior_KL(self):
        if self.whiten:
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_white_diag(self.q_mu, self.q_sqrt)
            else:
                KL = kullback_leiblers.gauss_kl_white(self.q_mu, self.q_sqrt)
        else:
            K = self.kern.K(self.Z) + eye(self.num_inducing) * settings.numerics.jitter_level
            if self.q_diag:
                KL = kullback_leiblers.gauss_kl_diag(self.q_mu, self.q_sqrt, K)
            else:
                KL = kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt, K)
        return KL

    def build_likelihood(self):
        """
        This gives a variational bound on the model likelihood.
        """

        # Get prior KL.
        KL = self.build_prior_KL()

        # Get conditionals
        fmean, fvar = self.build_predict(self.X, full_cov=False)

        # Get variational expectations.
        var_exp = self.likelihood.variational_expectations(fmean, fvar, self.Y)

        # re-scale for minibatch size
        scale = tf.cast(self.num_data, settings.dtypes.float_type) /\
            tf.cast(tf.shape(self.X)[0], settings.dtypes.float_type)

        return tf.reduce_sum(var_exp) * scale - KL

    def build_predict(self, Xnew, full_cov=False):
        mu, var = conditionals.conditional(Xnew, self.Z, self.kern, self.q_mu,
                                           q_sqrt=self.q_sqrt, full_cov=full_cov, whiten=self.whiten)
        return mu + self.mean_function(Xnew), var
