import numpy as np

def referenceRbfKernel( X, lengthScale, signalVariance ):
    (nDataPoints, inputDimensions ) = X.shape
    kernel = np.zeros( (nDataPoints, nDataPoints ) )
    for row_index in range( nDataPoints ):
        for column_index in range( nDataPoints ):
            vecA = X[row_index,:]
            vecB = X[column_index,:]
            delta = vecA - vecB
            distanceSquared = np.dot( delta.T, delta )
            kernel[row_index, column_index ] = signalVariance * np.exp( -0.5*distanceSquared / lengthScale** 2)
    return kernel


def referenceMLPKernel( X, weightVariances, biasVariance, signalVariance ):
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            enumerator = (weightVariances * x).dot(y) + biasVariance

            x_denominator = np.sqrt((weightVariances * x).dot(x) + biasVariance + 1.)
            y_denominator = np.sqrt((weightVariances * y).dot(y) + biasVariance + 1.)
            denominator = x_denominator * y_denominator

            kernel[row, col] = signalVariance * 2. / np.pi * np.arcsin(enumerator / denominator)
    return kernel


def referencePeriodicKernel( X, lengthScale, signalVariance, period ):
    # Based on the GPy implementation of standard_period kernel
    base = np.pi * (X[:, None, :] - X[None, :, :]) / period
    exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / lengthScale ), axis = -1 ) )
    return signalVariance * exp_dist
