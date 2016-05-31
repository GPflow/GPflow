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
