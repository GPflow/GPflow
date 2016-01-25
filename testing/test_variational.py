import GPflow
import tensorflow as tf
from GPflow.tf_hacks import eye
import numpy as np
import unittest
from IPython import embed

def referenceUnivariateLogMarginalLikelihood( y, K, noiseVariance ):
    return -0.5 * y * y / ( K + noiseVariance ) - 0.5 * np.log( K + noiseVariance ) - 0.5 * np.log( np.pi * 2. )
    
def referenceUnivariatePosterior( y, K, noiseVariance ):
    mean = K * y / ( K + noiseVariance )
    variance = K - K / ( K + noiseVariance )
    return mean, variance

def referenceUnivariatePriorKL( meanA, meanB, varA, varB ):
    #KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and qB are univariate normal distributions.
    return 0.5 * ( np.log( varB ) - np.log( varA) - 1. + varA/varB + (meanB-meanA) * (meanB - meanA) / varB )

def kernel(kernelVariance=1):
    kern = GPflow.kernels.White(1)
    kern.variance = kernelVariance
    return kern

class VariationalTest(unittest.TestCase):
    def setUp(self):
        #def __init__(self, X, Y, kern, likelihood, Z, mean_function=Zero(), num_latent=None, q_diag=False, whiten=True):
        self.y_real = 2.
        self.K = 1.
        self.noiseVariance = 0.5
        self.univariate = 1
        self.oneLatentFunction = 1
        self.meanZero = 0.
        self.X = np.atleast_2d( np.array( [0.] ) )
        self.Y = np.atleast_2d( np.array( [self.y_real] ) )
        self.Z = self.X.copy()
        self.lik = GPflow.likelihoods.Gaussian()
        self.lik.variance = self.noiseVariance
        self.posteriorMean, self.posteriorVariance = referenceUnivariatePosterior( y=self.y_real, K = self.K, noiseVariance = self.noiseVariance ) 
        self.posteriorStd = np.sqrt( self.posteriorVariance )

    
    def get_model( self, is_diagonal, is_whitened ):
        model = GPflow.svgp.SVGP( X=self.X, Y=self.Y, kern=kernel(kernelVariance=self.K), likelihood=self.lik, Z=self.Z, q_diag = is_diagonal, whiten = is_whitened ) 
        if is_diagonal:
            model.q_sqrt = np.ones((self.univariate, self.oneLatentFunction))*self.posteriorStd
        else:
            model.q_sqrt = np.ones( (self.univariate, self.univariate, self.oneLatentFunction ) ) * self.posteriorStd
        model.q_mu =  np.ones((self.univariate, self.oneLatentFunction))* self.posteriorMean
        model.make_tf_array(model._free_vars)    
        return model    
        
    def test_prior_KL( self ):
        meanA = self.posteriorMean
        varA = self.posteriorVariance
        meanB = self.meanZero #Assumes a zero 
        varB = self.K
        
        referenceKL = referenceUnivariatePriorKL( meanA, meanB, varA, varB )
        
        for is_diagonal in [True,False]:
            for is_whitened in [True,False]:        
                model = self.get_model( is_diagonal, is_whitened )
                with model.tf_mode():
                    prior_KL_function = model.build_prior_KL()
                test_prior_KL = prior_KL_function.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } ) 
                self.failUnless( np.abs( referenceKL - test_prior_KL ) < 1e-4 )
        
    def test_build_likelihood(self):
        #reference marginal likelihood
        log_marginal_likelihood = referenceUnivariateLogMarginalLikelihood( y=self.y_real, K = self.K, noiseVariance = self.noiseVariance ) 
        
        for is_diagonal in [True,False]:
            for is_whitened in [True,False]:
                model = self.get_model( is_diagonal, is_whitened )
                with model.tf_mode():
                    model_likelihood_function = model.build_likelihood()
                model_likelihood = model_likelihood_function.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } ) 
                self.failUnless( np.abs( model_likelihood - log_marginal_likelihood ) < 1e-4 )

if __name__ == "__main__":
    unittest.main()

