import gpflow
from gpflow.ci_utils import ci_niter
import tensorflow as tf
import numpy as np

nRepeats = ci_niter(50)

predict_limits = [-4., 4.]
inducing_points_limits = [-1., 9]
hold_out_limits = [0.20, 0.60]
optimization_limits = [18., 25.]

def readCsvFile(fileName):
    return np.loadtxt(fileName).reshape(-1, 1)

def getTrainingTestData():
    overallX = readCsvFile('data/snelson_train_inputs.dat')
    overallY = readCsvFile('data/snelson_train_outputs.dat')

    trainIndices = []
    testIndices = []

    nPoints = overallX.shape[0]

    for index in range(nPoints):
        if index % 4 == 0:
            trainIndices.append(index)
        else:
            testIndices.append(index)

    xtrain = overallX[trainIndices,:]
    xtest  = overallX[testIndices, :]
    ytrain = overallY[trainIndices, :]
    ytest  = overallY[testIndices, :]

    return xtrain, ytrain, xtest, ytest

def getLogPredictiveDensities(targetValues, means, variances):
    assert targetValues.shape == means.shape
    assert variances.shape == means.shape

    deltas = targetValues - means
    mahalanobisTerms = -0.5*deltas**2/variances
    normalizationTerms = -0.5 * np.log(variances) - 0.5 * np.log(2.*np.pi)
    return mahalanobisTerms + normalizationTerms

def getKernel():
    return gpflow.kernels.SquaredExponential()

def getRegressionModel(X, Y):
    m = gpflow.models.GPR(X, Y, kern=getKernel())
    m.likelihood.variance = 1.
    m.kern.lengthscales = 1.
    m.kern.variance = 1.
    return m

def getSparseModel(X, Y, isFITC=False):
    if isFITC:
        m = gpflow.models.GPRFITC((X, Y), kernel=getKernel(), inducing_variable=X.copy())
    else:
        m = gpflow.models.SGPR((X, Y), kernel=getKernel(), inducing_variable=X.copy())
    return m

def printModelParameters(model):
    print("  Likelihood variance = {:.5g}".format(model.likelihood.variance.numpy()))
    print("  Kernel variance     = {:.5g}".format(model.kernel.variance.numpy()))
    print("  Kernel lengthscale  = {:.5g}".format(model.kernel.lengthscale.numpy()))

def plotPredictions(ax, model, color, label=None):
    xtest = np.sort(readCsvFile('data/snelson_test_inputs.dat'))
    predMean, predVar = model.predict_y(xtest)
    ax.plot(xtest, predMean, color, label=label)
    ax.plot(xtest, predMean + 2.*np.sqrt(predVar), color)
    ax.plot(xtest, predMean - 2.*np.sqrt(predVar), color)

def repeatMinimization(model, xtest, ytest):
    callback = Callback(model, xtest, ytest)

    @tf.function(autograph=False)
    def objective_closure():
        return - model.log_marginal_likelihood()

    opt = gpflow.optimizers.Scipy()
    #print("Optimising for {} repetitions".format(nRepeats))
    for repeatIndex in range(nRepeats):
        #print(repeatIndex)
        opt.minimize(objective_closure, model.trainable_variables,
                     method='L-BFGS-B',
                     tol=1e-11,
                     options=dict(disp=False, maxiter=ci_niter(2000)),
                     step_callback=callback)
    return callback

def trainSparseModel(xtrain, ytrain, exact_model, isFITC, xtest, ytest):
    sparse_model = getSparseModel(xtrain, ytrain, isFITC)
    sparse_model.likelihood.variance = exact_model.likelihood.variance.read_value().copy()
    sparse_model.kern.lengthscales = exact_model.kern.lengthscales.read_value().copy()
    sparse_model.kern.variance = exact_model.kern.variance.read_value().copy()
    return sparse_model, repeatMinimization(sparse_model, xtest, ytest)

def plotComparisonFigure(xtrain, sparse_model,exact_model, ax_predictions, ax_inducing_points, ax_optimization, iterations, log_likelihoods,hold_out_likelihood, title=None):
    plotPredictions(ax_predictions, exact_model, 'g', label='Exact')
    plotPredictions(ax_predictions, sparse_model, 'b', label='Approximate')
    ax_predictions.legend(loc=9)
    ax_predictions.plot( sparse_model.inducing_variable.Z.numpy() , -1.*np.ones( xtrain.shape ), 'ko' )
    ax_predictions.set_ylim( predict_limits )
    ax_inducing_points.plot( xtrain, sparse_model.inducing_variable.Z.numpy(), 'bo' )
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

class Callback:
    def __init__(self, model, xtest, ytest, holdout_interval=10):
        self.model = model
        self.holdout_interval = holdout_interval
        self.xtest = xtest
        self.ytest = ytest
        self.log_likelihoods = []
        self.hold_out_likelihood = []
        self.n_iters = []
        self.counter = 0

    def __call__(self, step, variables, values):
        # step will reset to zero between calls to minimize(), whereas counter will keep increasing

        if (self.counter <= 10) or (self.counter % self.holdout_interval) == 0:
            for var, val in zip(variables, values):
                var.assign(val)

            self.n_iters.append(self.counter)
            self.log_likelihoods.append(self.model.log_likelihood().numpy())

            predictive_mean, predictive_variance = self.model.predict_y(self.xtest)
            test_log_likelihood = tf.reduce_mean(getLogPredictiveDensities(self.ytest, predictive_mean, predictive_variance))
            self.hold_out_likelihood.append(test_log_likelihood.numpy())

        self.counter+=1

def stretch(lenNIters, initialValues):
    stretched = np.ones(lenNIters) * initialValues[-1]
    stretched[0:len(initialValues)] = initialValues
    return stretched

def snelsonDemo():
    from matplotlib import pyplot as plt
    from IPython import embed
    xtrain,ytrain,xtest,ytest = getTrainingTestData()

    # run exact inference on training data.
    exact_model = getRegressionModel(xtrain,ytrain)
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(exact_model, maxiter=ci_niter(2000000))

    figA, axes = plt.subplots(1,1)
    inds = np.argsort(xtrain.flatten())
    axes.plot(xtrain[inds,:], ytrain[inds,:], 'ro')
    plotPredictions(axes, exact_model, 'g', None)

    figB, axes = plt.subplots(3,2)

    # run sparse model on training data initialized from exact optimal solution.
    VFEmodel, VFEcb = trainSparseModel(xtrain, ytrain, exact_model, False, xtest, ytest)
    FITCmodel, FITCcb = trainSparseModel(xtrain, ytrain, exact_model, True, xtest, ytest)

    print("Exact model parameters \n")
    printModelParameters(exact_model)
    print("Sparse model parameters for VFE optimization \n")
    printModelParameters(VFEmodel)
    print("Sparse model parameters for FITC optimization \n")
    printModelParameters(FITCmodel)

    VFEiters = FITCcb.n_iters
    VFElog_likelihoods = stretch(len(VFEiters), VFEcb.log_likelihoods)
    VFEhold_out_likelihood = stretch(len(VFEiters), VFEcb.hold_out_likelihood)

    plotComparisonFigure(xtrain, VFEmodel, exact_model, axes[0,0], axes[1,0], axes[2,0], VFEiters, VFElog_likelihoods, VFEhold_out_likelihood, "VFE")
    plotComparisonFigure(xtrain, FITCmodel, exact_model, axes[0,1], axes[1,1], axes[2,1],FITCcb.n_iters, FITCcb.log_likelihoods, FITCcb.hold_out_likelihood , "FITC")

    axes[0,0].set_title('VFE', loc='center', fontdict = {'fontsize': 22})
    axes[0,1].set_title('FITC', loc='center', fontdict = {'fontsize': 22})

    embed()


if __name__ == '__main__':
    snelsonDemo()
