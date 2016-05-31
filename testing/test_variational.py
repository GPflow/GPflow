import GPflow
import tensorflow as tf
from GPflow.tf_hacks import eye
import numpy as np
import unittest
from reference import *

def referenceUnivariateLogMarginalLikelihood( y, K, noiseVariance ):
    return -0.5 * y * y / ( K + noiseVariance ) - 0.5 * np.log( K + noiseVariance ) - 0.5 * np.log( np.pi * 2. )
    
def referenceUnivariatePosterior( y, K, noiseVariance ):
    mean = K * y / ( K + noiseVariance )
    variance = K - K / ( K + noiseVariance )
    return mean, variance

def referenceUnivariatePriorKL( meanA, meanB, varA, varB ):
    #KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and qB are univariate normal distributions.
    return 0.5 * ( np.log( varB ) - np.log( varA) - 1. + varA/varB + (meanB-meanA) * (meanB - meanA) / varB )

def referenceMultivariatePriorKL( meanA, covA, meanB, covB ):
    #KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and aB are K dimensional multivariate normal distributions.
    #Analytically tractable and equal to...
    #... 0.5 * ( Tr( covB^{-1} covA) + (meanB - meanA)^T covB^{-1} (meanB - meanA) - K + log( det( covB ) ) - log ( det( covA ) ) )
    K = covA.shape[0]
    traceTerm = 0.5 * np.trace( np.linalg.solve( covB, covA ) )
    delta = meanB - meanA 
    mahalanobisTerm = 0.5 * np.dot( delta.T, np.linalg.solve( covB, delta ) )
    constantTerm = -0.5 * K
    priorLogDeterminantTerm = 0.5*np.linalg.slogdet( covB )[1]
    variationalLogDeterminantTerm = -0.5 * np.linalg.slogdet( covA )[1] 
    return traceTerm + mahalanobisTerm + constantTerm + priorLogDeterminantTerm + variationalLogDeterminantTerm

def kernel(kernelVariance=1,lengthScale=1.):
    kern = GPflow.kernels.RBF(1)
    kern.variance = kernelVariance
    kern.lengthscales = lengthScale
    return kern

class VariationalUnivariateTest(unittest.TestCase):
    def setUp(self):
        tf.reset_default_graph()
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
                model._compile()
                with model.tf_mode():
                    model_likelihood_function = model.build_likelihood()
                model_likelihood = model_likelihood_function.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } ) 
                self.failUnless( np.abs( model_likelihood - log_marginal_likelihood ) < 1e-4 )

    def testUnivariateConditionals(self):
        for is_diagonal in [True,False]:
            for is_whitened in [True,False]:
                model = self.get_model( is_diagonal, is_whitened )
                with model.tf_mode():
                    if is_whitened:
                        fmean_func, fvar_func = GPflow.conditionals.gaussian_gp_predict_whitened(self.X, self.Z, model.kern, model.q_mu, model.q_sqrt, self.oneLatentFunction )
                    else:
                        fmean_func, fvar_func = GPflow.conditionals.gaussian_gp_predict(self.X, self.Z, model.kern, model.q_mu, model.q_sqrt, self.oneLatentFunction)  
                mean_value = fmean_func.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } )[0,0] 
                var_value = fvar_func.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } )[0,0] 
                self.failUnless( np.abs( mean_value - self.posteriorMean ) < 1e-4 )
                self.failUnless( np.abs( var_value - self.posteriorVariance ) < 1e-4 )

class VariationalMultivariateTest(unittest.TestCase):
    def setUp( self ):
        tf.reset_default_graph()
        self.nDimensions = 3
        self.rng = np.random.RandomState(1)
        self.Y = self.rng.randn( self.nDimensions, 1 )
        self.X = self.rng.randn( self.nDimensions, 1 )
        self.Z = self.X.copy()
        self.noiseVariance = 0.5
        self.signalVariance = 1.5
        self.lengthScale = 1.7
        self.oneLatentFunction = 1
        self.lik = GPflow.likelihoods.Gaussian()
        self.lik.variance = self.noiseVariance        
        self.q_mean = self.rng.randn( self.nDimensions, self.oneLatentFunction )
        self.q_sqrt_diag = self.rng.rand( self.nDimensions, self.oneLatentFunction )
        self.q_sqrt_full = np.tril( self.rng.rand( self.nDimensions, self.nDimensions ) )


    def getModel( self, is_diagonal, is_whitened  ):
        model = GPflow.svgp.SVGP( X=self.X, Y=self.Y, kern=kernel(kernelVariance=self.signalVariance,lengthScale=self.lengthScale), likelihood=self.lik, Z=self.Z, q_diag = is_diagonal, whiten = is_whitened ) 
        if is_diagonal:
            model.q_sqrt = self.q_sqrt_diag
        else:
            model.q_sqrt = self.q_sqrt_full[:,:,None]
        model.q_mu =  self.q_mean
        model.make_tf_array(model._free_vars)    
        return model 
    
    def test_refrence_implementation_consistency( self ):
        rng = np.random.RandomState(10)
        qMean = rng.randn()
        qCov = rng.rand()
        pMean = rng.rand()
        pCov = rng.rand()
        univariate_KL = referenceUnivariatePriorKL( qMean, pMean, qCov, pCov )
        multivariate_KL = referenceMultivariatePriorKL( np.array( [[qMean]] ), np.array( [[qCov]]), np.array( [[pMean]] ), np.array( [[pCov]] ) )  
        self.failUnless( np.abs( univariate_KL - multivariate_KL ) < 1e-4 )  
        
    def test_prior_KL_fullQ( self ):
        covQ = np.dot( self.q_sqrt_full, self.q_sqrt_full.T )
        mean_prior = np.zeros( ( self.nDimensions, 1 ) )

        for is_whitened in [True,False]:
            model = self.getModel( False, is_whitened )

            if is_whitened:
                cov_prior = np.eye( self.nDimensions )
            else:
                cov_prior = referenceRbfKernel( self.X, self.lengthScale, self.signalVariance )
        

            referenceKL = referenceMultivariatePriorKL( self.q_mean, covQ, mean_prior, cov_prior )
            #now get test KL.
            with model.tf_mode():
                prior_KL_function = model.build_prior_KL()
            test_prior_KL = prior_KL_function.eval( session = model._session, feed_dict = {model._free_vars: model.get_free_state() } ) 
            self.failUnless( np.abs( referenceKL - test_prior_KL ) < 1e-4 )        
        
if __name__ == "__main__":
    unittest.main()

