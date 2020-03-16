import numpy as np


def ref_rbf_kernel(X, lengthscales, signal_variance):
    N, _ = X.shape
    kernel = np.zeros((N, N))
    for row_index in range(N):
        for column_index in range(N):
            vecA = X[row_index, :]
            vecB = X[column_index, :]
            delta = vecA - vecB
            distance_squared = np.dot(delta.T, delta)
            kernel[row_index, column_index] = signal_variance * np.exp(
                -0.5 * distance_squared / lengthscales ** 2
            )
    return kernel


def ref_arccosine_kernel(X, order, weight_variances, bias_variance, signal_variance):
    num_points = X.shape[0]
    kernel = np.empty((num_points, num_points))
    for row in range(num_points):
        for col in range(num_points):
            x = X[row]
            y = X[col]

            numerator = (weight_variances * x).dot(y) + bias_variance

            x_denominator = np.sqrt((weight_variances * x).dot(x) + bias_variance)
            y_denominator = np.sqrt((weight_variances * y).dot(y) + bias_variance)
            denominator = x_denominator * y_denominator

            theta = np.arccos(np.clip(numerator / denominator, -1.0, 1.0))
            if order == 0:
                J = np.pi - theta
            elif order == 1:
                J = np.sin(theta) + (np.pi - theta) * np.cos(theta)
            elif order == 2:
                J = 3.0 * np.sin(theta) * np.cos(theta)
                J += (np.pi - theta) * (1.0 + 2.0 * np.cos(theta) ** 2)

            kernel[row, col] = (
                signal_variance
                * (1.0 / np.pi)
                * J
                * x_denominator ** order
                * y_denominator ** order
            )
    return kernel


def ref_periodic_kernel(X, base_name, lengthscales, signal_variance, period):
    """
    Calculates K(X) for the periodic kernel based on various base kernels.
    """
    sine_arg = np.pi * (X[:, None, :] - X[None, :, :]) / period
    sine_base = np.sin(sine_arg) / lengthscales
    if base_name in {"RBF", "SquaredExponential"}:
        dist = 0.5 * np.sum(np.square(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern12":
        dist = np.sum(np.abs(sine_base), axis=-1)
        exp_dist = np.exp(-dist)
    elif base_name == "Matern32":
        dist = np.sqrt(3) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist) * np.exp(-dist)
    elif base_name == "Matern52":
        dist = np.sqrt(5) * np.sum(np.abs(sine_base), axis=-1)
        exp_dist = (1 + dist + dist ** 2 / 3) * np.exp(-dist)
    return signal_variance * exp_dist
