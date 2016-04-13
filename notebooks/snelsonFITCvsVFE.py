import GPflow
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
import cProfile
import csv

def readCsvFile( fileName ):
    reader = csv.reader(open(fileName,'r') )
    dataList = []
    for row in reader:
        dataList.append( [float(elem) for elem in row ] )
    
    return np.array( dataList )
    
def getTrainingTestData():
    overallX = readCsvFile( 'train_inputs' )
    overallY = readCsvFile( 'train_outputs' )
    
    trainIndeces = []
    testIndeces = []
    
    nPoints = overallX.shape[0]
    
    for index in range(nPoints):
        if ( (index%4) == 0):
            trainIndeces.append( index )
        else:
            testIndeces.append( index )
            
    xtrain = overallX[ trainIndeces,: ]
    xtest = overallX[ testIndeces, : ]
    ytrain = overallY[ trainIndeces, : ]
    ytest  = overallY[ testIndeces, : ]
    
    return xtrain,ytrain,xtest,ytest
    
def getPredictPoints():
    predpoints = readCsvFile( 'test_inputs' )
    return predpoints

def getKernel():
    return GPflow.kernels.RBF(1)

def getRegressionModel(X,Y):
    m = GPflow.gpr.GPR(X, Y, kern=getKernel() )
    m.likelihood.variance = 1.
    m.kern.lengthscales = 1.
    m.kern.variance = 1.
    return m
    
def getSparseModel(X,Y,isFITC=False):
    if not(isFITC):
        m = GPflow.sgpr.SGPR(X, Y, kern=getKernel(),  Z=X.copy() )
    else:
        print "here "
        m = GPflow.sgpr.GPRFITC(X, Y, kern=getKernel(),  Z=X.copy() )
    return m

def printModelParameters( model ):
    print "Likelihood variance ", model.likelihood.variance, "\n"
    print "Kernel variance ", model.kern.variance, "\n"
    print "Kernel lengthscale ", model.kern.lengthscales, "\n"

def plotPredictions( model, color ):
    xtest = readCsvFile( 'test_inputs' )
    predMean, predVar = model.predict_y(xtest)
    plt.plot( xtest, predMean, color )
    plt.plot( xtest, predMean + 2.*np.sqrt(predVar),color )
    plt.plot( xtest, predMean - 2.*np.sqrt(predVar), color )

def getInitialVariationalCovariance( sparse_model ):
    gramMatrix = sparse_model.K( sparse_model.X )
    
    
    return q_mean, q_std

def trainSparseModel(xtrain,ytrain,exact_model,isFITC):
    sparse_model = getSparseModel(xtrain,ytrain,True)
    sparse_model.likelihood.variance._array = exact_model.likelihood.variance._array
    sparse_model.kern.lengthscales._array = exact_model.kern.lengthscales._array
    sparse_model.kern.variance._array = exact_model.kern.variance._array

    sparse_model.optimize(max_iters = 200000  )
    return sparse_model    

def plotComparisonFigure(xtrain, sparse_model,exact_model, figure_number, title ):
    plt.figure(figure_number)
    plotPredictions( exact_model, 'go' )
    plotPredictions( sparse_model, 'bo' )
    plt.plot( xtrain, np.ones( xtrain.shape ), 'ko' )
    plt.plot( sparse_model.Z._array , -1.*np.ones( xtrain.shape ), 'ko' )
    plt.title(title)

def snelsonDemo():
    xtrain,ytrain,xtest,ytest = getTrainingTestData()
    
    #run exact inference on training data.
    exact_model = getRegressionModel(xtrain,ytrain)
        
    exact_model.optimize(max_iters = 200000 )
   
    #run sparse model on training data intialized from exact optimal solution.
    VFEmodel = trainSparseModel(xtrain,ytrain,exact_model,False)
    FITCmodel = trainSparseModel(xtrain,ytrain,exact_model,False)

    print "Exact model parameters \n"
    printModelParameters( exact_model )
    print "Sparse model parameters for VFE optimization \n"
    printModelParameters( VFEmodel )
    print "Sparse model parameters for FITC optimization \n"
    printModelParameters( FITCmodel )
    
    plotComparisonFigure( xtrain, FITCmodel, exact_model, 1, "FITC comparison" )
    plotComparisonFigure( xtrain, VFEmodel, exact_model, 2, "VFE comparison" )
    
    embed()
    
if __name__ == '__main__':
    snelsonDemo()
    
