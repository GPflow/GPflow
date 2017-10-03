"""
Demos using random features to approximate draws from a GP.
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy import linalg as sla
import gpflow


NUM_OBS = 200
AXIS_EDGE = 0.5
NOISE_STD = 0.05

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



def create_approx_features_func(model, data_dim):

    tf_graph = tf.Graph()
    tf_sess = tf.Session(graph=tf_graph)

    with tf_graph.as_default():
        x_ph = tf.placeholder(tf.float64, [None, data_dim])

        with model.kern.tf_mode():
            x_free = tf.placeholder('float64')
            model.kern.make_tf_array(x_free)
            feats_func = model.kern.create_feature_map_func(model.random_seed_for_random_features)
            feats = feats_func(x_ph)


    def approx_features_func(X):
        feats_evald = tf_sess.run(feats, feed_dict={x_ph:X, x_free: model.kern.get_free_state()})
        return feats_evald

    return approx_features_func




def main2():

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    import copy
    num_inducing = 20
    inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    model = gpflow.svgp.SVGP(data['x_data'], data['y'], kern=kernel, Z=copy.copy(data['x_data'][inducing_starts, :]),
                             likelihood=gpflow.likelihoods.Gaussian(), whiten=False)

    model.optimize()



    mean, var = model.predict_y(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)

    approx_features_func = create_approx_features_func(model, x_data.shape[1])
    #mean, L_precision = model.linear_weights_posterior()
    mean = model.linear_weights_posterior()
    #L_var += 1 * np.eye(L_var.shape[0])
    #L_var = sla.cholesky(L_var, lower=True)
    for j in range(2):
        #sampled_var = sla.solve_triangular(L_precision.T, rng.randn(mean.shape[0]), lower=False)[:, None]
        #sampled_var = (L_var @ rng.randn(mean.shape[0]))[:, None]

        theta_sample = mean #+ sampled_var


        phi_at_xindcs = approx_features_func(x_axis_points)
        f_sample = phi_at_xindcs @ theta_sample
        plt.plot(x_axis_points, f_sample, '--', color='#727882')


    for m in model.predict_f_samples(x_axis_points, 2):
        plt.plot(x_axis_points, m, 'gray')


    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass


    plt.show()


def main():

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    import copy
    num_inducing = 20
    inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    model = gpflow.svgp.SVGP(data['x_data'], data['y'], kern=kernel, Z=copy.copy(data['x_data'][inducing_starts, :]),
                             likelihood=gpflow.likelihoods.Gaussian(), whiten=True)

    model.optimize()



    mean, var = model.predict_y(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)

    approx_features_func = create_approx_features_func(model, x_data.shape[1])
    #mean, L_precision = model.linear_weights_posterior()
    mean1, mean2, mean3, mean4 = model.linear_weights_posterior()
    #L_var += 1 * np.eye(L_var.shape[0])
    #L_var = sla.cholesky(L_var, lower=True)

    phi_at_xindcs = approx_features_func(x_axis_points)
    f_sample = phi_at_xindcs @ mean1
    plt.plot(x_axis_points, f_sample, '-', lw=2, color='#2fb1fc', label="mean1")

    f_sample = phi_at_xindcs @ mean2
    plt.plot(x_axis_points, f_sample, ':', lw=3.5, color='#f21f96', label="mean2")

    phi_at_xindcs = approx_features_func(x_axis_points)
    f_sample = phi_at_xindcs @ mean3
    plt.plot(x_axis_points, f_sample, '--', lw=2, color='#e08002', label="mean5")

    f_sample = phi_at_xindcs @ mean4
    plt.plot(x_axis_points, f_sample, '-.', lw=2.5, color='#6ff252', label="mean6")

    # #
    # f_sample = phi_at_xindcs @ mean3
    # plt.plot(x_axis_points, f_sample, '--', lw=2, color='#e000c2', label="mean3")




    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.legend()
    plt.show()


def main3():

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    import copy
    num_inducing = 20
    inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    model = gpflow.svgp.SVGP(data['x_data'], data['y'], kern=kernel, Z=copy.copy(data['x_data'][inducing_starts, :]),
                             likelihood=gpflow.likelihoods.Gaussian(), whiten=True)

    model.optimize()



    mean, var = model.predict_f(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, var[:, 0], 'b', lw=2)


    approx_features_func = create_approx_features_func(model, x_data.shape[1])
    L_var1, L_var2 = model.linear_weights_posterior_variance_play()



    phi_at_xindcs = approx_features_func(x_axis_points)
    f_sample = phi_at_xindcs @ L_var1 @ phi_at_xindcs.T
    plt.plot(x_axis_points, np.diag(f_sample), '-', lw=2, color='#e07b00', label="var1")

    f_sample = phi_at_xindcs @ L_var2 @ phi_at_xindcs.T
    plt.plot(x_axis_points, np.diag(f_sample), ':', lw=3, color='#00e01d', label="var2")



    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.legend()
    plt.show()




def main4():

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    import copy
    num_inducing = 20
    inducing_starts = rng.choice(data['x_data'].shape[0], num_inducing, replace=False)
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    model = gpflow.svgp.SVGP(data['x_data'], data['y'], kern=kernel, Z=copy.copy(data['x_data'][inducing_starts, :]),
                             likelihood=gpflow.likelihoods.Gaussian(), whiten=True)

    model.optimize()

    mean, var = model.predict_f(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)



    approx_features_func = create_approx_features_func(model, x_data.shape[1])
    mean1, var1, mean2, precchol2 = model.linear_weights_posterior_methods_play()

    phi_at_xindcs = approx_features_func(x_axis_points)
    f_sample = phi_at_xindcs @ mean1
    plt.plot(x_axis_points, f_sample, '-', lw=2, color='#e07b00', label="mean_orig")

    f_sample = phi_at_xindcs @ mean2
    plt.plot(x_axis_points, f_sample, ':', lw=3, color='#00e01d', label="mean_new")



    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.legend()
    plt.show()



def gpr_marginal_vars():

    rng = np.random.RandomState(100)
    x_data, x_axis_points = create_random_x_data(rng)
    data = package_data(x_data, x_axis_points, rng)

    kernel = gpflow.kernels.RBF(1)
    kernel.lengthscales = 0.1
    #model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)
    import copy

    model = gpflow.gpr.GPR(data['x_data'], data['y'], kern=kernel)


    model.optimize()



    mean, var = model.predict_f(x_axis_points)
    f, ax = plt.subplots()
    ax.plot(x_data, data['f'], 'kx', mew=2)
    ax.plot(x_axis_points, var[:, 0], 'b', lw=2)


    approx_features_func = create_approx_features_func(model, x_data.shape[1])
    mean, chol = model.linear_weights_posterior()



    phi_at_xindcs = approx_features_func(x_axis_points)
    #f_sample = phi_at_xindcs @ L_var1 @ phi_at_xindcs.T

    half = sla.solve_triangular(chol, phi_at_xindcs.T, lower=True)
    var = half.T @ half

    plt.plot(x_axis_points, np.diag(var), '-', lw=2, color='#e07b00', label="var1")




    try:
        inducing_values = model.Z.value
        plt.plot(inducing_values, np.zeros_like(inducing_values), 'bo')
    except AttributeError:
        pass

    plt.legend()
    plt.show()



if __name__ == '__main__':
    main3()


