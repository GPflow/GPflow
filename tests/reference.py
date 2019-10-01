import numpy as np


def referenceRbfKernel(X, lengthScale, signalVariance):
    nDataPoints, _ = X.shape
    kernel = np.zeros((nDataPoints, nDataPoints))
    for row_index in range(nDataPoints):
        for column_index in range(nDataPoints):
            vecA = X[row_index,:]
            vecB = X[column_index,:]
            delta = vecA - vecB
            distanceSquared = np.dot(delta.T, delta)
            kernel[row_index, column_index] = signalVariance * np.exp(-0.5 * distanceSquared / lengthScale**2)
    return kernel


def referenceArcCosineKernel(X, order, weightVariances, biasVariance, signalVariance):
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weightVariances * x).dot(y) + biasVariance

            x_denominator = np.sqrt((weightVariances * x).dot(x) + biasVariance)
            y_denominator = np.sqrt((weightVariances * y).dot(y) + biasVariance)
            denominator = x_denominator * y_denominator

            theta = np.arccos(np.clip(numerator / denominator, -1., 1.))
            if order == 0:
                J = np.pi - theta
            elif order == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            elif order == 2:
                J = 3. * np.sin(theta) * np.cos(theta)
                J += (np.pi - theta) * (1. + 2. * np.cos(theta) ** 2)

            kernel[row, col] = signalVariance * (1. / np.pi) * J * \
                               x_denominator ** order * \
                               y_denominator ** order
    return kernel


def referencePeriodicKernel(X, lengthScale, signalVariance, period, baseClassName="RBF"):
    # Based on the GPy implementation of standard_period kernel
    lengthScale = np.array(lengthScale)
    base = np.pi * (X[:, None, :] - X[None, :, :]) / period
    sine_base = np.sin(base) / lengthScale[None, None, :]
    if baseClassName in {"RBF", "SquaredExponential"}:
        dist = 0.5 * np.sum(np.square(sine_base), axis=-1)
        K = np.exp(-dist)
    elif baseClassName == "Matern12":
        dist = np.sum(np.abs(sine_base), axis=-1)
        K = np.exp(-dist)
    elif baseClassName == "Matern32":
        dist = np.sqrt(3) * np.sum(np.abs(sine_base), axis=-1)
        K = (1 + dist) * np.exp(-dist)
    elif baseClassName == "Matern52":
        dist = np.sqrt(5) * np.sum(np.abs(sine_base), axis=-1)
        K = (1 + dist + dist ** 2 / 3) * np.exp(-dist)
    return signalVariance * K
