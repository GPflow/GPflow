from __future__ import print_function
import gpflow
import tensorflow as tf
import os
import numpy as np
import cProfile
import csv

tol=1e-11
nRepeats = 50

predict_limits = [-4., 4.]
inducing_points_limits = [-1., 9]
hold_out_limits = [0.20, 0.60]
optimization_limits = [18., 25.]

def readCsvFile(fileName):
    reader = csv.reader(open(fileName,'r'))
    dataList = []
    for row in reader:
        dataList.append([float(elem) for elem in row])

    return np.array(dataList)

def getTrainingTestData():
    overallX = readCsvFile('data/snelson_train_inputs')
    overallY = readCsvFile('data/snelson_train_outputs')

    trainIndeces = []
    testIndeces = []

    nPoints = overallX.shape[0]

    for index in range(nPoints):
        if index % 4 == 0:
            trainIndeces.append(index)
        else:
            testIndeces.append(index)

    xtrain = overallX[trainIndeces,:]
    xtest = overallX[testIndeces, :]
    ytrain = overallY[trainIndeces, :]
    ytest  = overallY[testIndeces, :]

    return xtrain,ytrain,xtest,ytest

def getLogPredictiveDensities(targetValues, means, variances):
    assert(targetValues.flatten().shape == targetValues.shape)
    assert(means.flatten().shape == means.shape)
    assert(variances.flatten().shape == variances.shape)

    assert(len(targetValues) == len(means))
    assert(len(variances) == len(means))

    deltas = targetValues - means
    mahalanobisTerms = -0.5*deltas**2/variances
    normalizationTerms = -0.5 * np.log(variances) - 0.5 * np.log(2.*np.pi)
    return mahalanobisTerms + normalizationTerms

def getKernel():
    return gpflow.kernels.RBF(1)

def getRegressionModel(X,Y):
    m = gpflow.models.GPR(X, Y, kern=getKernel())
    m.likelihood.variance = 1.
    m.kern.lengthscales = 1.
    m.kern.variance = 1.
    return m

def getSparseModel(X,Y,isFITC=False):
    if not(isFITC):
        m = gpflow.models.SGPR(X, Y, kern=getKernel(),  Z=X.copy())
    else:
        m = gpflow.models.GPRFITC(X, Y, kern=getKernel(),  Z=X.copy())
    return m

def printModelParameters(model):
    print("Likelihood variance ", model.likelihood.variance, "\n")
    print("Kernel variance ", model.kern.variance, "\n")
    print("Kernel lengthscale ", model.kern.lengthscales, "\n")

def plotPredictions(ax, model, color, label):
    xtest = np.sort(readCsvFile('data/snelson_test_inputs'))
    predMean, predVar = model.predict_y(xtest)
    ax.plot(xtest, predMean, color, label=label)
    ax.plot(xtest, predMean + 2.*np.sqrt(predVar),color)
    ax.plot(xtest, predMean - 2.*np.sqrt(predVar), color)

def trainSparseModel(xtrain,ytrain,exact_model,isFITC, xtest, ytest):
    sparse_model = getSparseModel(xtrain,ytrain,isFITC)
    sparse_model.likelihood.variance = exact_model.likelihood.variance.read_value().copy()
    sparse_model.kern.lengthscales = exact_model.kern.lengthscales.read_value().copy()
    sparse_model.kern.variance = exact_model.kern.variance.read_value().copy()
    callback = cb(sparse_model, xtest, ytest)
    for repeatIndex in range(nRepeats):
        print("repeatIndex ",repeatIndex)
        sparse_model.optimize(disp=False, maxiter= 2000, tol=tol, callback=callback)
    return sparse_model, callback

def plotComparisonFigure(xtrain, sparse_model,exact_model, ax_predictions, ax_inducing_points, ax_optimization, iterations, log_likelihoods,hold_out_likelihood, title):
    plotPredictions(ax_predictions, exact_model, 'g', label='Exact')
    plotPredictions(ax_predictions, sparse_model, 'b', label='Approximate')
    ax_predictions.legend(loc=9)
    ax_predictions.plot( sparse_model.feature.Z.value , -1.*np.ones( xtrain.shape ), 'ko' )
    ax_predictions.set_ylim( predict_limits )
    ax_inducing_points.plot( xtrain, sparse_model.feature.Z.value, 'bo' )
    xs= np.linspace( ax_inducing_points.get_xlim()[0], ax_inducing_points.get_xlim()[1], 200 )
    ax_inducing_points.plot( xs, xs, 'g' )
    ax_inducing_points.set_xlabel('Optimal inducing point position')
    ax_inducing_points.set_ylabel('Learnt inducing point position')
    ax_inducing_points.set_ylim(inducing_points_limits)
    ax_optimization.plot(iterations, -1.*np.array(log_likelihoods), 'g-')
    ax_optimization.set_ylim(optimization_limits)
    ax2 = ax_optimization.twinx()
    ax2.plot(iterations, -1.*np.array(hold_out_likelihood), 'b-')
    ax_optimization.set_xlabel('Minimization iterations')
    ax_optimization.set_ylabel('Minimization objective', color='g')
    ax2.set_ylim(hold_out_limits)
    ax2.set_ylabel('Hold out negative log likelihood', color='b')

class cb():
    def __init__(self, model, xtest, ytest, holdout_inverval=100):
        self.model = model
        self.holdout_inverval = holdout_inverval
        self.xtest = xtest
        self.ytest = ytest
        self.log_likelihoods = []
        self.hold_out_likelihood = []
        self.n_iters = []
        self.counter = 0

    def __call__(self, info):
        if (self.counter%self.holdout_inverval) == 0 or (self.counter <= 10):
            predictive_mean, predictive_variance = self.model.predict_y(self.xtest)
            self.n_iters.append(self.counter)
            self.log_likelihoods.append(self.model.compute_log_likelihood())
            self.hold_out_likelihood.append(getLogPredictiveDensities(self.ytest.flatten() , predictive_mean.flatten(), predictive_variance.flatten()).mean())
        self.counter+=1

def stretch(lenNIters, initialValues):
    stretched = np.ones(lenNIters) * initialValues[-1]
    stretched[0:len(initialValues)] = initialValues
    return stretched

def snelsonDemo():
    from matplotlib import pyplot as plt
    from IPython import embed
    xtrain,ytrain,xtest,ytest = getTrainingTestData()

    #run exact inference on training data.
    exact_model = getRegressionModel(xtrain,ytrain)
    exact_model.optimize(maxiter= 2000000, tol=tol)

    figA, axes = plt.subplots(1,1)
    inds = np.argsort(xtrain.flatten())
    axes.plot(xtrain[inds,:], ytrain[inds,:], 'ro')
    plotPredictions(axes, exact_model, 'g', None)

    figB, axes = plt.subplots(3,2)

    #run sparse model on training data intialized from exact optimal solution.
    VFEmodel, VFEcb = trainSparseModel(xtrain,ytrain,exact_model,False,xtest,ytest)
    FITCmodel, FITCcb = trainSparseModel(xtrain,ytrain,exact_model,True,xtest,ytest)

    print("Exact model parameters \n")
    printModelParameters(exact_model)
    print("Sparse model parameters for VFE optimization \n")
    printModelParameters(VFEmodel)
    print("Sparse model parameters for FITC optimization \n")
    printModelParameters(FITCmodel)

    VFEiters = FITCcb.n_iters
    VFElog_likelihoods = stretch(len(VFEiters), VFEcb.log_likelihoods)
    VFEhold_out_likelihood = stretch(len(VFEiters), VFEcb.hold_out_likelihood)

    plotComparisonFigure(xtrain, VFEmodel, exact_model, axes[0,0], axes[1,0], axes[2,0], VFEiters, VFElog_likelihoods.tolist(), VFEhold_out_likelihood.tolist(), "VFE")
    plotComparisonFigure(xtrain, FITCmodel, exact_model, axes[0,1], axes[1,1], axes[2,1],FITCcb.n_iters, FITCcb.log_likelihoods, FITCcb.hold_out_likelihood , "FITC")

    axes[0,0].set_title('VFE', loc='center', fontdict = {'fontsize': 22})
    axes[0,1].set_title('FITC', loc='center', fontdict = {'fontsize': 22})

    embed()


if __name__ == '__main__':
    snelsonDemo()
