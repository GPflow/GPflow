import gpflow
from gpflow.ci_utils import ci_niter
import tensorflow as tf
import numpy as np

nRepeats = ci_niter(50)

predict_limits = [-4.0, 4.0]
inducing_points_limits = [-1.0, 9]
hold_out_limits = [0.20, 0.60]
optimization_limits = [18.0, 25.0]


def readCsvFile(fileName):
    return np.loadtxt(fileName).reshape(-1, 1)


def getTrainingTestData():
    overallX = readCsvFile("data/snelson_train_inputs.dat")
    overallY = readCsvFile("data/snelson_train_outputs.dat")

    trainIndices = []
    testIndices = []

    nPoints = overallX.shape[0]

    for index in range(nPoints):
        if index % 4 == 0:
            trainIndices.append(index)
        else:
            testIndices.append(index)

    Xtrain = overallX[trainIndices, :]
    Xtest = overallX[testIndices, :]
    Ytrain = overallY[trainIndices, :]
    Ytest = overallY[testIndices, :]

    return Xtrain, Ytrain, Xtest, Ytest


def getLogPredictiveDensities(targetValues, means, variances):
    assert targetValues.shape == means.shape
    assert variances.shape == means.shape

    deltas = targetValues - means
    mahalanobisTerms = -0.5 * deltas ** 2 / variances
    normalizationTerms = -0.5 * np.log(variances) - 0.5 * np.log(2.0 * np.pi)
    return mahalanobisTerms + normalizationTerms


def getKernel():
    return gpflow.kernels.SquaredExponential()


def getRegressionModel(X, Y):
    m = gpflow.models.GPR(X, Y, kern=getKernel())
    m.likelihood.variance = 1.0
    m.kern.lengthscales = 1.0
    m.kern.variance = 1.0
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
    print("  Kernel lengthscale  = {:.5g}".format(model.kernel.lengthscales.numpy()))


def plotPredictions(ax, model, color, label=None):
    Xtest = np.sort(readCsvFile("data/snelson_test_inputs.dat"))
    predMean, predVar = model.predict_y(Xtest)
    ax.plot(Xtest, predMean, color, label=label)
    ax.plot(Xtest, predMean + 2.0 * np.sqrt(predVar), color)
    ax.plot(Xtest, predMean - 2.0 * np.sqrt(predVar), color)


def repeatMinimization(model, Xtest, Ytest):
    callback = Callback(model, Xtest, Ytest)

    opt = gpflow.optimizers.Scipy()
    # print("Optimising for {} repetitions".format(nRepeats))
    for repeatIndex in range(nRepeats):
        # print(repeatIndex)
        opt.minimize(
            model.training_loss,
            model.trainable_variables,
            method="L-BFGS-B",
            tol=1e-11,
            options=dict(disp=False, maxiter=ci_niter(2000)),
            step_callback=callback,
            compile=True,
        )
    return callback


def trainSparseModel(Xtrain, Ytrain, exact_model, isFITC, Xtest, Ytest):
    sparse_model = getSparseModel(Xtrain, Ytrain, isFITC)
    sparse_model.likelihood.variance = exact_model.likelihood.variance.numpy()
    sparse_model.kern.lengthscales = exact_model.kern.lengthscales.numpy()
    sparse_model.kern.variance = exact_model.kern.variance.numpy()
    return sparse_model, repeatMinimization(sparse_model, Xtest, Ytest)


def plotComparisonFigure(
    Xtrain,
    sparse_model,
    exact_model,
    ax_predictions,
    ax_inducing_points,
    ax_optimization,
    iterations,
    log_likelihoods,
    hold_out_likelihood,
    title=None,
):
    plotPredictions(ax_predictions, exact_model, "g", label="Exact")
    plotPredictions(ax_predictions, sparse_model, "b", label="Approximate")
    ax_predictions.legend(loc=9)
    ax_predictions.plot(
        sparse_model.inducing_variable.Z.numpy(), -1.0 * np.ones(Xtrain.shape), "ko"
    )
    ax_predictions.set_ylim(predict_limits)
    ax_inducing_points.plot(Xtrain, sparse_model.inducing_variable.Z.numpy(), "bo")
    xs = np.linspace(ax_inducing_points.get_xlim()[0], ax_inducing_points.get_xlim()[1], 200)
    ax_inducing_points.plot(xs, xs, "g")
    ax_inducing_points.set_xlabel("Optimal inducing point position")
    ax_inducing_points.set_ylabel("Learnt inducing point position")
    ax_inducing_points.set_ylim(inducing_points_limits)
    ax_optimization.plot(iterations, -1.0 * np.array(log_likelihoods), "g-")
    ax_optimization.set_ylim(optimization_limits)
    ax2 = ax_optimization.twinx()
    ax2.plot(iterations, -1.0 * np.array(hold_out_likelihood), "b-")
    ax_optimization.set_xlabel("Minimization iterations")
    ax_optimization.set_ylabel("Minimization objective", color="g")
    ax2.set_ylim(hold_out_limits)
    ax2.set_ylabel("Hold out negative log likelihood", color="b")


class Callback:
    def __init__(self, model, Xtest, Ytest, holdout_interval=10):
        self.model = model
        self.holdout_interval = holdout_interval
        self.Xtest = Xtest
        self.Ytest = Ytest
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
            self.log_likelihoods.append(self.model.maximum_log_likelihood_objective().numpy())

            predictive_mean, predictive_variance = self.model.predict_y(self.Xtest)
            test_log_likelihood = tf.reduce_mean(
                getLogPredictiveDensities(self.Ytest, predictive_mean, predictive_variance)
            )
            self.hold_out_likelihood.append(test_log_likelihood.numpy())

        self.counter += 1


def stretch(lenNIters, initialValues):
    stretched = np.ones(lenNIters) * initialValues[-1]
    stretched[0 : len(initialValues)] = initialValues
    return stretched


def snelsonDemo():
    from matplotlib import pyplot as plt
    from IPython import embed

    Xtrain, Ytrain, Xtest, Ytest = getTrainingTestData()

    # run exact inference on training data.
    exact_model = getRegressionModel(Xtrain, Ytrain)
    opt = gpflow.train.ScipyOptimizer()
    opt.minimize(exact_model, maxiter=ci_niter(2000000))

    figA, axes = plt.subplots(1, 1)
    inds = np.argsort(Xtrain.flatten())
    axes.plot(Xtrain[inds, :], Ytrain[inds, :], "ro")
    plotPredictions(axes, exact_model, "g", None)

    figB, axes = plt.subplots(3, 2)

    # run sparse model on training data initialized from exact optimal solution.
    VFEmodel, VFEcb = trainSparseModel(Xtrain, Ytrain, exact_model, False, Xtest, Ytest)
    FITCmodel, FITCcb = trainSparseModel(Xtrain, Ytrain, exact_model, True, Xtest, Ytest)

    print("Exact model parameters \n")
    printModelParameters(exact_model)
    print("Sparse model parameters for VFE optimization \n")
    printModelParameters(VFEmodel)
    print("Sparse model parameters for FITC optimization \n")
    printModelParameters(FITCmodel)

    VFEiters = FITCcb.n_iters
    VFElog_likelihoods = stretch(len(VFEiters), VFEcb.log_likelihoods)
    VFEhold_out_likelihood = stretch(len(VFEiters), VFEcb.hold_out_likelihood)

    plotComparisonFigure(
        Xtrain,
        VFEmodel,
        exact_model,
        axes[0, 0],
        axes[1, 0],
        axes[2, 0],
        VFEiters,
        VFElog_likelihoods,
        VFEhold_out_likelihood,
        "VFE",
    )
    plotComparisonFigure(
        Xtrain,
        FITCmodel,
        exact_model,
        axes[0, 1],
        axes[1, 1],
        axes[2, 1],
        FITCcb.n_iters,
        FITCcb.log_likelihoods,
        FITCcb.hold_out_likelihood,
        "FITC",
    )

    axes[0, 0].set_title("VFE", loc="center", fontdict={"fontsize": 22})
    axes[0, 1].set_title("FITC", loc="center", fontdict={"fontsize": 22})

    embed()


if __name__ == "__main__":
    snelsonDemo()
