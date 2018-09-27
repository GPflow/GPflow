# Unmaintained demo script
# see the GP Regression notebook in the GPflow documentation,
# https://gpflow.readthedocs.io/en/develop/notebooks/regression.html

import gpflow
import tensorflow as tf
import os
import numpy as np

def getData():
    rng = np.random.RandomState(1)
    N = 30
    X = rng.rand(N,1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + rng.randn(N,1)*0.1 + 3
    return X, Y

def getRegressionModel(X, Y):
    #build the GPR object
    k = gpflow.kernels.Matern52(1)
    meanf = gpflow.mean_functions.Linear(1,0)
    m = gpflow.models.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01
    print("Here are the parameters before optimization")
    m
    return m

def optimizeModel(m):
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(m)
    print("Here are the parameters after optimization")
    m

def setModelPriors(m):
    #we'll choose rather arbitrary priors.
    m.clear()
    m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
    m.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.likelihood.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.mean_function.A.prior = gpflow.priors.Gaussian(0., 10.)
    m.mean_function.b.prior = gpflow.priors.Gaussian(0., 10.)
    m.compile()
    print("model with priors ", m)

def getSamples(m):
    hmc = gpflow.train.HMC()
    samples = hmc.sample(m, num_samples=100, epsilon = 0.1)
    return samples

def runExperiments(sampling=True,outputGraphs=False):
    X,Y = getData()
    m = getRegressionModel(X,Y)
    optimizeModel(m)
    if sampling:
        setModelPriors(m)
        samples = getSamples(m)

if __name__ == '__main__':
    runExperiments()
    # cProfile.run('runExperiments(plotting=False)')
