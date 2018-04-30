import tensorflow as tf
from gpflow import settings,  densities, transforms, kullback_leiblers, features, conditionals
from gpflow.core.compilable import Build
from gpflow.params import Parameterized, ParamList, DataHolder, Parameter, Minibatch, DataHolder
from gpflow.decors import autoflow,params_as_tensors, params_as_tensors_for
from gpflow.mean_functions import Zero
from gpflow.models import Model
from gpflow.likelihoods import HeteroscedasticLikelihood
from gpflow.latent import Latent
from gpflow import autoflow
float_type = settings.float_type

class HGP(Model):
    """
    The base class for Deep Gaussian process models.
    Implements a Monte-Carlo variational bound and convenience functions.
    """
    def __init__(self, X, Y, Z, kern, likelihood, 
                 mean_function=Zero, 
                 minibatch_size=None,
                 num_latent = None, 
                 num_samples=1,
                 num_data=None,
                 whiten=True):
        Model.__init__(self)
        self.num_samples = num_samples
        self.num_latent = num_latent or Y.shape[1]
        self.num_data = num_data or X.shape[0]
            
        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = likelihood
        assert isinstance(likelihood,HeteroscedasticLikelihood)
        
        self.f_latent = Latent(Z, mean_function, kern, num_latent=num_latent, 
                               whiten=whiten, name="f_latent")

    @params_as_tensors
    def _build_predict(self, X, full_cov=False, S=1):
        if S is not None:
            X = tf.tile(X[None,:,:],[S,1,1])
        return self.f_latent.conditional(X, full_cov=full_cov)
    
    @params_as_tensors
    def _build_predict_sample(self, X, full_cov=False, S=1):
        if S is not None:
            X = tf.tile(X[None,:,:],[S,1,1])
        return self.f_latent.sample_from_conditional(X, full_cov=full_cov)

    def E_log_p_Y(self, X, Y):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        X = tf.tile(X[None,:,:],[self.num_samples,1,1])
        Fmean, Fvar = self._build_predict(X, full_cov=False, S=None)
        hetero_variance = tf.square(self.likelihood.hetero_noise(X))
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y, hetero_variance)  # S, N, D
        return tf.reduce_mean(var_exp, 0)  # N, D

    @params_as_tensors
    def KL_tensors(self):
        KL = [self.f_latent.KL()]
        if hasattr(self.likelihood,'log_noise_latent'):
            KL.append(self.likelihood.log_noise_latent.KL())
        if hasattr(self.f_latent.kern,'log_ls_latent'):
            KL.append(self.f_latent.kern.log_ls_latent.KL())
        if hasattr(self.f_latent.kern,'log_sigma_latent'):
            KL.append(self.f_latent.kern.log_sigma_latent.KL())
        return KL
    
    @params_as_tensors
    def _build_likelihood(self):
        L = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y))
        KL = tf.reduce_sum([self.KL_tensors()])
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return L * scale - KL     
        
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=False, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_f_full_cov(self, Xnew, num_samples):
        return self._build_predict(Xnew, full_cov=True, S=num_samples)

    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        """
        Draws the predictive mean and variance at the points `X`
        num_samples times.
        X should be [N,D] and this returns [S,N,num_latent], [S,N,num_latent]
        """
        Xnew = tf.tile(Xnew[None,:,:],[num_samples,1,1])
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=None)
        hetero_variance = tf.square(self.likelihood.hetero_noise(Xnew))
        return self.likelihood.predict_mean_and_var(Fmean, Fvar, hetero_variance)

    @autoflow((float_type, [None, None]), (float_type, [None, None]), (tf.int32, []))
    def predict_density(self, Xnew, Ynew, num_samples):
        Xnew = tf.tile(Xnew[None,:,:],[num_samples,1,1])
        Fmean, Fvar = self._build_predict(Xnew, full_cov=False, S=None)
        hetero_variance = tf.square(self.likelihood.hetero_variance(Xnew))
        l = self.likelihood.predict_density(Fmean, Fvar, Ynew, hetero_variance)
        log_num_samples = tf.log(tf.cast(num_samples, float_type))
        return tf.reduce_logsumexp(l - log_num_samples, axis=0)
