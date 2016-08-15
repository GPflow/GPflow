import GPflow
import tensorflow as tf
import os
import numpy as np

def getData():
    rng = np.random.RandomState( 1 )
    N = 30
    X = rng.rand(N,1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + rng.randn(N,1)*0.1 + 3
    return X,Y
    
def getRegressionModel(X,Y):
    #build the GPR object
    k = GPflow.kernels.Matern52(1)
    meanf = GPflow.mean_functions.Linear(1,0)
    m = GPflow.gpr.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01
    print "Here are the parameters before optimization"
    m
    return m

def optimizeModel(m):
    m.optimize()
    print "Here are the parameters after optimization"
    m

def setModelPriors( m ):
    #we'll choose rather arbitrary priors. 
    m.kern.lengthscales.prior = GPflow.priors.Gamma(1., 1.)
    m.kern.variance.prior = GPflow.priors.Gamma(1., 1.)
    m.likelihood.variance.prior = GPflow.priors.Gamma(1., 1.)
    m.mean_function.A.prior = GPflow.priors.Gaussian(0., 10.)
    m.mean_function.b.prior = GPflow.priors.Gaussian(0., 10.)
    print "model with priors ", m

def getSamples( m ):
    samples = m.sample(100, epsilon = 0.1)
    return samples

def runExperiments(plotting=True,outputGraphs=False):
    X,Y = getData()
    m = getRegressionModel(X,Y)
    optimizeModel(m)
    setModelPriors( m )
    samples = getSamples( m )

if __name__ == '__main__':
    runExperiments()
    #cProfile.run( 'runExperiments( plotting=False )' )
