import numpy as np


def ref_rbf_kernel(X, lengthscale, signal_var):
    N, _ = X.shape
    kernel = np.zeros((N, N))
    for row_index in range(N):
        for column_index in range(N):
            vecA = X[row_index, :]
            vecB = X[column_index, :]
            delta = vecA - vecB
            distance_squared = np.dot(delta.T, delta)
            kernel[row_index, column_index] = \
                signal_var * np.exp(-0.5 * distance_squared / lengthscale ** 2)
    return kernel


def ref_arccosine_kernel(X, order, weightVariances, biasVariance, signal_var):
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weightVariances * x).dot(y) + biasVariance

            x_denominator = np.sqrt((weightVariances * x).dot(x) +
                                    biasVariance)
            y_denominator = np.sqrt((weightVariances * y).dot(y) +
                                    biasVariance)
            denominator = x_denominator * y_denominator

            theta = np.arccos(np.clip(numerator / denominator, -1., 1.))
            if order == 0:
                J = np.pi - theta
            elif order == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            elif order == 2:
                J = 3. * np.sin(theta) * np.cos(theta)
                J += (np.pi - theta) * (1. + 2. * np.cos(theta)**2)

            kernel[row, col] = signal_var * (1. / np.pi) * J * \
                x_denominator ** order * \
                y_denominator ** order
    return kernel


def ref_periodic_kernel(X, lengthscale, signal_var, period):
    # Based on the GPy implementation of standard_period kernel
    base = np.pi * (X[:, None, :] - X[None, :, :]) / period
    exp_dist = np.exp(-0.5 *
                      np.sum(np.square(np.sin(base) / lengthscale), axis=-1))
    return signal_var * exp_dist
