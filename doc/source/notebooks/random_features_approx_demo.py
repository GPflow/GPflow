"""
Demos using random features to approximate draws from a GP.

This file has two main demos.
- draw_samples: will draw samples from the GP and from the GP via the random features approximation
 you can guess which are which!
- thompson_sample_min: this shows how random features can be used to perform efficient Thompson
sampling of the minima of functions.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import linalg as sla
from scipy import optimize

import gpflow
from gpflow import decors


NUM_OBS = 200
AXIS_EDGE = 0.5
NOISE_STD = 0.05
SVGP = True  # switch between using GPR or SVGP.


def func1(x):
    f = np.sin(12 * x) + 0.66 * np.cos(25 * x)
    deriv = 12 * np.cos(12 * x) - 0.66 * 25 * np.sin(25 * x)  # the analytical deriv
    second_deriv = -144 * np.sin(12 * x) - 412.5 * np.cos(25 * x)
    return f, deriv, second_deriv


def create_random_x_data(rng):
    x_data = rng.rand(NUM_OBS, 1)  # make it as a column vector
    x_axis_indcs = np.linspace(x_data.min() - AXIS_EDGE, x_data.max() + AXIS_EDGE, 150)[:, None]
    return x_data, x_axis_indcs


def package_data(x_data, x_axis_idcs, rng):
    f_data, derivs, second_derivs = func1(x_data)

    data = dict(
        x_axis=x_axis_idcs,
        x_data=x_data,

        f=f_data,
        deriv=derivs,
        second_deriv=second_derivs
    )
    data['y'] = data['f'] + NOISE_STD * rng.randn(*data['f'].shape)
    return data


def draw_samples():
    """
    Train a GP and then use the random features approximations to draw samples. Try to spot the
    difference between samples generated this way and samples generated via the true Kernel!
    """

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1

    if SVGP:
        num_inducing = 30
        inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
        model = gpflow.models.SVGP(data['x_data'], data['y'], kern=kernel,
                                 Z=np.copy(data['x_data'][inducing_starts, :]),
                                 likelihood=gpflow.likelihoods.Gaussian(), whiten=True)
    else:
        model = gpflow.models.GPR(data['x_data'], data['y'], kern=kernel)

    model.compile()
    optimizer = gpflow.train.ScipyOptimizer(options=dict(maxiter=100))
    optimizer.minimize(model)

    mean, var = model.predict_y(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)

    def approx_features_func(X):
        return model.kern.feature_map(X)

    mean, precision_or_var, var_flag = model.linear_weights_posterior()
    L = sla.cholesky(precision_or_var, lower=True)

    for j in range(2):

        if var_flag:
            sampled_var = (L @ rng.randn(mean.shape[0]))[:, None]
        else:
            sampled_var = sla.solve_triangular(L.T, rng.randn(mean.shape[0]), lower=False)[:, None]

        theta_sample = mean + sampled_var

        phi_at_xindcs = approx_features_func(x_axis_points)
        f_sample = phi_at_xindcs @ theta_sample
        plt.plot(x_axis_points, f_sample, '--', color='#727882', label="approx_{}".format(j))

    for i, m in enumerate(model.predict_f_samples(x_axis_points, 2)):
        plt.plot(x_axis_points, m, 'gray', label="actual_{}".format(i))

    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.show()


def _create_approx_func_with_grads(model, data_dim, theta):
    """
    This provides a function that
    evaluates the sampled function's (defined by the provided theta) value and the gradient
    of this value with respect to the inputs.
    """
    tf_graph = model.graph
    tf_sess = model.session

    with tf_graph.as_default():
        x_ph = tf.placeholder(tf.float64, [None, data_dim])

        previous_unitialised_vars = decors._get_set_of_unit_var_names(model.session)
        feats = model.kern._feature_map(x_ph)
        func = tf.matmul(feats, theta)
        func_deriv = tf.gradients(func, x_ph)

        # We then see if this has created any new variables and initialise these.
        current_unitialised_vars = decors._get_set_of_unit_var_names(model.session)
        model.session.run(tf.variables_initializer(
            decors._collect_vars_with_name(current_unitialised_vars - previous_unitialised_vars)
        ))


    def approx_func(X):
        X = np.atleast_2d(X)
        fd = {x_ph:X}
        if model.feeds:
            fd.update(model.feeds)
        func_evald, deriv_evald = tf_sess.run([func, func_deriv],
                                              feed_dict=fd)
        return func_evald[0].astype(np.float64), deriv_evald[0].astype(np.float64)

    return approx_func


def thompson_sample_min():
    """
    Show how Thompson Sampling can be used using the random features to sample from
    a continuous domain.
    """
    PLOT_EACH_SAMPLE = False

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1

    if SVGP:
        num_inducing = 30
        inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
        model = gpflow.models.SVGP(data['x_data'], data['y'], kern=kernel,
                                 Z=np.copy(data['x_data'][inducing_starts, :]),
                                 likelihood=gpflow.likelihoods.Gaussian(), whiten=True)
    else:
        model = gpflow.models.GPR(data['x_data'], data['y'], kern=kernel)

    model.compile()
    optimizer = gpflow.train.ScipyOptimizer(options=dict(maxiter=100))
    optimizer.minimize(model)

    mean, var = model.predict_y(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)


    mean, precision_or_var, var_flag = model.linear_weights_posterior()
    L = sla.cholesky(precision_or_var, lower=True)

    for j in range(10):

        if var_flag:
            sampled_var = (L @ rng.randn(mean.shape[0]))[:, None]
        else:
            sampled_var = sla.solve_triangular(L.T, rng.randn(mean.shape[0]), lower=False)[:, None]

        theta_sample = mean + sampled_var

        f_sample = _create_approx_func_with_grads(model, x_data.shape[1], theta_sample)

        initial_sample = np.array([rng.choice(x_data.flatten())])[:, None]
        opt_res = optimize.minimize(lambda x: f_sample(x), x0=initial_sample, method="L-BFGS-B",
                                    jac=True, bounds=[(x_axis_points.min(),
                                                      x_axis_points.max())])

        x = opt_res.x
        f = f_sample(opt_res.x)[0]

        if PLOT_EACH_SAMPLE:
            phi_at_xindcs = model.kern.feature_map(x_axis_points)
            sample = phi_at_xindcs @ theta_sample
            plt.plot(x_axis_points, sample)

        plt.plot(x, f, 'o', color='#8ae082', label="min_{}".format(j))

        if PLOT_EACH_SAMPLE:
            plt.plot(initial_sample, func1(initial_sample)[0], 'o', color='#e84560', label="intial")
            plt.legend()
            plt.show()

    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.show()


if __name__ == '__main__':
    draw_samples()


