from matplotlib import pyplot as plt
import gpflow
import tensorflow as tf
import os
import numpy as np
import cProfile

def outputGraph(model, dirName, fileName):
    model.compile()
    if not(os.path.isdir(dirName)):
        os.mkdir(dirName)
    fullFileName = os.path.join(dirName, fileName)
    if os.path.isfile(fullFileName):
        os.remove(fullFileName)
    tf.train.write_graph(model.session.graph_def, dirName+'/', fileName, as_text=False)

# build a very simple data set:
def getData():
    rng = np.random.RandomState(1)
    N = 30
    X = rng.rand(N,1)
    Y = np.sin(12*X) + 0.66*np.cos(25*X) + rng.randn(N,1)*0.1 + 3
    return X,Y

def plotData(X,Y):
    plt.figure()
    plt.plot(X, Y, 'kx', mew=2)

def getRegressionModel(X,Y):
    #build the GPR object
    k = gpflow.kernels.Matern52(1)
    meanf = gpflow.mean_functions.Linear(1,0)
    m = gpflow.models.GPR(X, Y, k, meanf)
    m.likelihood.variance = 0.01
    print "Here are the parameters before optimization"
    m
    return m

def optimizeModel(m):
    m.optimize()
    print "Here are the parameters after optimization"
    m

def plotOptimizationResult(X,Y,m):
    #plot!
    xx = np.linspace(-0.1, 1.1, 100)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure()
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    plt.plot(xx, mean + 2*np.sqrt(var), 'b--', xx, mean - 2*np.sqrt(var), 'b--', lw=1.2)

def setModelPriors(m):
    #we'll choose rather arbitrary priors. 
    m.kern.lengthscales.prior = gpflow.priors.Gamma(1., 1.)
    m.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.likelihood.variance.prior = gpflow.priors.Gamma(1., 1.)
    m.mean_function.A.prior = gpflow.priors.Gaussian(0., 10.)
    m.mean_function.b.prior = gpflow.priors.Gaussian(0., 10.)
    print "model with priors ", m

def getSamples(m):
    samples = m.sample(100, epsilon = 0.1)
    return samples

def plotSamples(X, Y, m, samples):
    xx = np.linspace(-0.1, 1.1, 100)[:,None]
    plt.figure()
    plt.plot(samples)

    f, axs = plt.subplots(1,3, figsize=(12,4), tight_layout=True)
    axs[0].plot(samples[:,0], samples[:,1], 'k.', alpha = 0.15)
    axs[0].set_xlabel('noise_variance')
    axs[0].set_ylabel('signal_variance')
    axs[1].plot(samples[:,0], samples[:,2], 'k.', alpha = 0.15)
    axs[1].set_xlabel('noise_variance')
    axs[1].set_ylabel('lengthscale')
    axs[2].plot(samples[:,2], samples[:,1], 'k.', alpha = 0.1)
    axs[2].set_xlabel('lengthscale')
    axs[2].set_ylabel('signal_variance')

    #an attempt to plot the function posterior
    #Note that we should really sample the function values here, instead of just using the mean.
    #We are under-representing the uncertainty here.
    # TODO: get full_covariance of the predictions (predict_f only?)

    plt.figure()

    for s in samples:
        m.set_state(s)
        mean, _ = m.predict_y(xx)
        plt.plot(xx, mean, 'b', lw=2, alpha = 0.05)

    plt.plot(X, Y, 'kx', mew=2)

def showAllPlots():
    plt.show()

def runExperiments(plotting=True,outputGraphs=False):
    X,Y = getData()
    if plotting:
        plotData(X,Y)
    m = getRegressionModel(X,Y)
    if outputGraphs:
        modelDir = 'models'
        outputGraph(m, modelDir, 'pointHypers')
    optimizeModel(m)
    if plotting:
        plotOptimizationResult(X,Y,m)
    setModelPriors(m)
    if outputGraphs:
        outputGraph(m, modelDir, 'bayesHypers')
    samples = getSamples(m)
    if plotting:
        plotSamples(X, Y, m,  samples)
        showAllPlots()

if __name__ == '__main__':
    runExperiments()
    #cProfile.run('runExperiments(plotting=False)')
