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


def referenceArcCosineKernel( X, order, num_layers, weightVariances, biasVariance, signalVariance ):
    def J(theta):
        if order == 0:
            return np.pi - theta
        elif order == 1:
            return np.sin(theta) + (np.pi - theta) * np.cos(theta)
        elif order == 2:
            return  3. * np.sin(theta) * np.cos(theta) + \
                    (np.pi - theta) * (1. + 2. * np.cos(theta) ** 2)

    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weightVariances * x).dot(y) + biasVariance

            x_denominator = np.sqrt((weightVariances * x).dot(x) + biasVariance)
            y_denominator = np.sqrt((weightVariances * y).dot(y) + biasVariance)

            xy_theta = np.arccos(np.clip(numerator / x_denominator / y_denominator,
                                         -1., 1.))
            xy_kernel = (1. / np.pi) * J(xy_theta) * \
                        x_denominator ** order * \
                        y_denominator ** order

            xx_kernel = (1. / np.pi) * J(0.) * x_denominator ** order
            yy_kernel = (1. / np.pi) * J(0.) * y_denominator ** order

            for _ in range(num_layers - 1):
                xy_theta = np.arccos(np.clip(xy_kernel / \
                                             np.sqrt(xx_kernel * yy_kernel),
                                             -1., 1.))

                xy_kernel = (1. / np.pi) * J(xy_theta) * \
                            (xx_kernel * yy_kernel) ** (0.5 * order)

                xx_kernel = (1. / np.pi) * J(0.) * xx_kernel ** order
                yy_kernel = (1. / np.pi) * J(0.) * yy_kernel ** order

            kernel[row, col] = signalVariance * xy_kernel
    return kernel


def referencePeriodicKernel( X, lengthScale, signalVariance, period ):
    # Based on the GPy implementation of standard_period kernel
    base = np.pi * (X[:, None, :] - X[None, :, :]) / period
    exp_dist = np.exp( -0.5* np.sum( np.square(  np.sin( base ) / lengthScale ), axis = -1 ) )
    return signalVariance * exp_dist
