"""
This file demonstrates training a Gaussian Process with derivative observations.
"""
import time
import copy
import collections

import numpy as np
import matplotlib.pyplot as plt

import gpflow


MAX_MODEL_ITERS = 50
NUM_OBS = 25
SET_VARIANCE = False
AXIS_EDGE = 0.5


def time_preds(model, x_axis_points):
    time_before = time.time()
    mean, var = model.predict_y(x_axis_points)
    time_for_predictions = time.time() - time_before
    return mean, var, time_for_predictions


def plot(ax, mean, var, x_axis_points, x_observations, y_observations,
         do_derivs=False, do_second_derivs=False):
    
    ax.plot(x_axis_points, func1(x_axis_points)[0], 'g', lw=4)
    ax.plot(x_observations, y_observations, 'kx', mew=2)

    f, derivs, second_derivs = func1(x_observations)

    ax.plot(x_axis_points, mean, 'b', lw=2)
    ax.fill_between(x_axis_points[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)
    if do_derivs:
        for loc, f_i, der in zip(x_observations, f, derivs):
            dx = 0.05
            ax.arrow(loc[0], f_i[0], dx, der[0] * dx)
    if do_second_derivs:
        for loc, f_i, s_der in zip(x_observations, f, second_derivs):
            dx = 0.05
            ax.arrow(loc[0], f_i[0], dx, s_der[0] * dx, color='r')

    return ax


def func1(x):
    f = np.sin(12*x) + 0.66*np.cos(25*x)
    deriv = 12 * np.cos(12*x) - 0.66 * 25 * np.sin(25*x)  # the analytical deriv
    second_deriv = -144 * np.sin(12*x) - 412.5 * np.cos(25*x)
    return f, deriv, second_deriv


def create_random_x_data(rng):
    x_data = rng.rand(NUM_OBS, 1)  # make it as a column vector
    x_axis_indcs = np.linspace(x_data.min() - AXIS_EDGE, x_data.max() + AXIS_EDGE, 150)[:, None]
    return x_data, x_axis_indcs
    
    
def package_data(x_data, x_axis_idcs):

    f_data, derivs, second_derivs = func1(x_data)

    x_axis_for_deriv_f_and_deriv = np.concatenate((
        np.tile(x_data, [2, 1]), np.concatenate([np.zeros_like(x_data), np.ones_like(x_data)],axis=0)
    ), axis=1)
    x_axis_for_deriv_f_d_and_sd = np.concatenate((
        x_axis_for_deriv_f_and_deriv, np.concatenate(
            [x_data, np.ones_like(x_data)*2.], axis=1
        )
    ), axis=0)

    data = dict(
        x_axis_plain=x_axis_idcs,
        x_axis_for_deriv_models=np.concatenate((x_axis_idcs, np.zeros_like(x_axis_idcs)), axis=1),

        x_plain=x_data,
        x_for_deriv_f_only=np.concatenate((x_data, np.zeros_like(x_data)), axis=1),
        x_for_deriv_f_and_deriv=x_axis_for_deriv_f_and_deriv,
        x_for_deriv_f_and_both_derivs=x_axis_for_deriv_f_d_and_sd,
        f=f_data,
        f_and_d=np.vstack([f_data, derivs]),
        f_and_d_and_sd=np.vstack([f_data, derivs, second_derivs])
    )
    return data



def make_models_and_package_with_correct_axis(data):
    input_dim = 1
    obs_dim = 1

    # Plain model rbf
    rbf_kern1 = gpflow.kernels.RBF(input_dim)
    model_no_derivs = gpflow.gpr.GPR(data['x_plain'], data['f'], kern=rbf_kern1)

    # Derivative dynamic kernel with gradient obs
    rbf_kern2 = gpflow.kernels.RBF(input_dim)
    kern2 = gpflow.derivative_kernel.DifferentialObservationsKernelDynamic(input_dim, rbf_kern2, obs_dim)
    model_deriv_dyn_d = gpflow.gpr.GPR(data['x_for_deriv_f_and_deriv'], data['f_and_d'], kern=kern2)

    # RBF deriv with no gradient observations
    rbf_kern3 = gpflow.kernels.RBF(input_dim)
    kern3 = gpflow.derivative_kernel.derivative_kernel_factory(input_dim, obs_dim, rbf_kern3)
    model_deriv_rbf =  gpflow.gpr.GPR(data['x_for_deriv_f_only'], data['f'], kern=kern3)

    # RBF deriv with deriv information
    rbf_kern4 = gpflow.kernels.RBF(input_dim)
    kern4 = gpflow.derivative_kernel.derivative_kernel_factory(input_dim, obs_dim, rbf_kern4)
    model_deriv_rbf_d = gpflow.gpr.GPR(data['x_for_deriv_f_and_deriv'], data['f_and_d'], kern=kern4)

    # RBF deriv with deriv and second deriv information
    rbf_kern5 = gpflow.kernels.RBF(input_dim)
    kern5 = gpflow.derivative_kernel.derivative_kernel_factory(input_dim, obs_dim, rbf_kern5)
    model_deriv_rbf_d_and_sd = gpflow.gpr.GPR(data['x_for_deriv_f_and_both_derivs'], data['f_and_d_and_sd'], kern=kern5)

    # model, data for making predictions, whether uses  deriv, whether uses first deriv, add callback
    models = collections.OrderedDict([
        ("Plain RBF model", (model_no_derivs, data['x_axis_plain'], False, False, False)),
        ("Dynamic deriv. model (f, df)",
            (model_deriv_dyn_d, data['x_axis_for_deriv_models'], True, False, True)),
        ("RBF deriv. model (f)",
         (model_deriv_rbf, data['x_axis_for_deriv_models'], False, False, False)),
        ("RBF deriv. model (f, df)",
         (model_deriv_rbf_d, data['x_axis_for_deriv_models'], True, False, False)),
        ("RBF deriv. model (f, df, d^2f)",
         (model_deriv_rbf_d_and_sd, data['x_axis_for_deriv_models'], True, True, False))
    ])

    if SET_VARIANCE:
        for model, *_ in models.values():
            print("Setting (and fixing) the variance of all models' likelihood small")
            model.likelihood.variance = 1e-3
            model.likelihood.variance.fixed = True

    return models


def main():
    rng = np.random.RandomState(100)
    x_data, x_axis_indcs = create_random_x_data(rng)

    data = package_data(x_data, x_axis_indcs)

    packaged_models = make_models_and_package_with_correct_axis(data)

    f, axarr = plt.subplots(2, 3, figsize=(15,12))
    flat_ax = axarr.flatten()

    for (ax, (name, (model, x_axis_data, do_derivs, do_sderivs, callback))) in zip(flat_ax, packaged_models.items()):
        print("Training {}...".format(name))
        time_before = time.time()
        if callback:
            # some models are very slow so useful to see actually doing something!
            callback = lambda x: print("step!")
            model.optimize(maxiter=MAX_MODEL_ITERS, callback=callback)
        else:
            model.optimize(maxiter=MAX_MODEL_ITERS)
        print("training took: {}".format(time.time()-time_before))
        mean, var, time_predcs = time_preds(model, x_axis_data)
        print("predictions took: {}".format(time_predcs))
        plot(ax, mean, var, data['x_axis_plain'], data['x_plain'], data['f'], do_derivs, do_sderivs)
        ax.set_title(name + "\n prediction time: {}".format(time_predcs))
        print("================= \n\n")

    plt.show()



def single_deriv_observation():
    x_in = np.array([[0.,1]])
    y = np.array([[1.]])
    rbf_kern = gpflow.kernels.RBF(1)
    kern = gpflow.derivative_kernel.derivative_kernel_factory(1, 1, rbf_kern)
    model_deriv_rbf =  gpflow.gpr.GPR(x_in, y, kern=kern)
    model_deriv_rbf.likelihood.variance = 1e-3

    x = np.linspace(-10, 10, 100)[:, None]
    x_data = np.hstack([x, np.zeros_like(x)])
    mean, var = model_deriv_rbf.predict_y(x_data)
    plt.plot(x, mean, 'b', lw=2)
    plt.fill_between(x[:, 0], mean[:, 0] - 2 * np.sqrt(var[:, 0]),
                    mean[:, 0] + 2 * np.sqrt(var[:, 0]), color='blue', alpha=0.2)

    for m in model_deriv_rbf.predict_f_samples(x_data, 5):
        plt.plot(x, m)

    plt.show()










if __name__ == '__main__':
    main()
    print("Done!")
