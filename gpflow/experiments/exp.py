
import numpy as np
import gpflow
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

np.random.seed(0)

plt.style.use('ggplot')


from gpflow.models.sparsegp import SparseVariationalMeanFieldGPs, SparseVariationalCoupledGPs


import pandas as pd

coupled = False

# #=================================================================================================
#
# # rho ~ (f1(x1) + 1)  * f2(x2)  + f3(x3)
# # rho ~ f1(x1)  * f2(x2)  + f3(x3)
# # paradigm parameters
#
# # --------------- model
xrange1 = [0, 2]
f1 = lambda x: np.exp(x / 2) -1.
xrange2 = [0, np.pi]
f2 = lambda x: 1 + np.cos(2.*x + np.pi / 3)
xrange3 =[0, 2]
f3 = lambda x: -np.sin(x)
fs = [f1, f2, f3]
xranges = [xrange1, xrange2, xrange3]
C = 3
#

maxiter = 8000
num_samples = 20
learning_rate = 1e-3
M = 60

for N in [50, 200, 500]:
    for coupled in [True]:



        observations = 'poisson'
        XFY = pd.read_csv('/home/vincent.adam/git/GPflow/gpflow/experiments/data.csv').to_numpy()
        X, F, Y = XFY[:N, slice(0,3)], XFY[:N, slice(3,6)], XFY[:N, slice(6,9)]
        #rho = (f1(X[:, 0]) + f2(X[:, 1]) * f3(X[:, 2]))[..., None] # predictor
        rho = (f1(X[:, 0]) * f2(X[:, 1]) + f3(X[:, 2]))[..., None] # predictor

        F = np.vstack([fs[c](X[:, c]) for c in range(C)]).T

        N = len(Y)
        data = (tf.constant(X), tf.constant(Y))
        # #=================================================================================================


        cols = 'rgb'

        # ## Building the model


        from gpflow.kernels import SquaredExponential, ConditionedKernel, Periodic

        kernels = [
            SquaredExponential(active_dims=[0], variance=1., lengthscale=1.),
            SquaredExponential(active_dims=[1], variance=1., lengthscale=.5),
            SquaredExponential(active_dims=[2], variance=1., lengthscale=1.5)
        ]
#        gpflow.utilities.set_trainable(kernels[1].period, False)

        # for k in kernels:
        #     gpflow.utilities.set_trainable(k, False)

        from gpflow.mean_functions import Zero

        mean_functions = [Zero() for _ in range(C)]

        indices = [slice(c, c + 1) for c in range(C)]

        Zs = [
            np.linspace(X[:, c].min(), X[:, 0].max(), M).reshape(-1, 1).copy() + np.zeros((1, C)) for c in
            range(C)
        ]

        q_mus = [np.zeros((len(Zs[c]), 1)) for c in range(C)]

        if observations == 'binomial':
            likelihood = gpflow.likelihoods.Bernoulli()
        elif observations == 'poisson':
            likelihood = gpflow.likelihoods.Poisson()
        elif observations == 'gaussian':
            likelihood = gpflow.likelihoods.Gaussian(variance=.1)

        offsets_y = [np.array([[0.]]), np.array([[1.5]]), np.array([[0.]])]
        offsets_x = [np.zeros((1, C)) for _ in range(C)]

        if coupled:

            m = SparseVariationalCoupledGPs(kernels, likelihood, Zs, num_data=N, whiten=True,
                                            offsets_x=offsets_x,
                                            offsets_y=offsets_y,
                                            deterministic_optimisation=False,
                                            mean_functions=mean_functions,
                                            num_samples=num_samples)
            for o in m.q.offsets_x:
                o.trainable = False
            m.q._offsets_y[0].trainable = False
            m.q._offsets_y[1].trainable = False
            m.q._offsets_y[2].trainable = False
            #for iv in m.q.inducing_variables:
            #    gpflow.utilities.set_trainable(iv, False)

        else:

            m = SparseVariationalMeanFieldGPs(kernels, likelihood, Zs, num_data=N,
                                              deterministic_optimisation=False,
                                              mean_functions=mean_functions,
                                              offsets_x=offsets_x,
                                              offsets_y=offsets_y, num_samples=num_samples)

            gpflow.utilities.set_trainable(m.q.qs[0].offset_x, False)
            gpflow.utilities.set_trainable(m.q.qs[1].offset_x, False)
            gpflow.utilities.set_trainable(m.q.qs[2].offset_x, False)

            gpflow.utilities.set_trainable(m.q.qs[0].offset_y, False)
            gpflow.utilities.set_trainable(m.q.qs[1].offset_y, False)
            gpflow.utilities.set_trainable(m.q.qs[2].offset_y, False)
            #for q in m.q.qs:
            #    gpflow.utilities.set_trainable(q.inducing_variable, False)

    # 10 starting points
    # 0.1, pi/20, 0.1
        #for k in m.kernels:
        #    gpflow.utilities.set_trainable(k.lengthscale, False)

        log_likelihood = tf.function(autograph=False)(m.log_likelihood)

        # We turn off training for inducing point locations



        def run_adam(model, iterations):
            """
            Utility function running the Adam optimizer

            :param model: GPflow model
            :param interations: number of iterations
            """
            # Create an Adam Optimizer action
            logf = []
            adam = tf.optimizers.Adam(learning_rate=learning_rate)

            signature = ((
                tf.TensorSpec(data[0].shape, dtype=data[0].dtype), tf.TensorSpec(data[1].shape, dtype=data[1].dtype)
            ),)
            @tf.function(autograph=False, input_signature=signature)
            def optimization_step(batch):
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(model.trainable_variables)
                    objective = - model.elbo(batch)
                    grads = tape.gradient(objective, model.trainable_variables)
                adam.apply_gradients(zip(grads, model.trainable_variables))
                return objective

            for step in range(iterations):
                elbo = - optimization_step(data)
                if step % 10 == 0:
                    logf.append(elbo.numpy())
                    print(step, elbo.numpy())

            return logf


        logf = run_adam(m, maxiter)

        f_means, f_vars = m.q.predict_fs(X, full_output_cov=False)

        f_means, f_vars = f_means.numpy(), f_vars.numpy()
        f_stds = np.sqrt(f_vars)

        # fig, axarr = plt.subplots(3, 1, figsize=(5, 15))
        # axarr = axarr.flatten()
        # ax = axarr[0]
        # ax.plot(np.arange(maxiter)[::10], logf)
        # ax.set_xlabel('iteration')
        # ax.set_ylabel('ELBO');
        #
        # ax = axarr[1]
        # for c in range(C):
        #     o = np.argsort(X[:, c])
        #     ax.plot(X[o, c], F[o, c], '--', color=cols[c], mew=2)
        #     ax.fill_between(X[o, c],
        #                     f_means[o, c] - 2. * f_stds[o, c],
        #                     f_means[o, c] + 2. * f_stds[o, c], alpha=.2, facecolor=cols[c])
        #     ax.plot(X[o, c], f_means[o, c], '-', color=cols[c], mew=2)
        #
        # # print(m.inducing_variables[0].Z)
        #
        # # ax = axarr[2]
        # # ax.errorbar(rho, p_mean, yerr=np.sqrt(p_var), color='b', fmt='o')
        # # ax.plot(rho, rho, 'k-', alpha=.1)
        # # ax.set_xlabel('data rho')
        # # ax.set_ylabel('prediction');
        #
        # plt.suptitle('N=%d - %s' % (N, observations))
        # plt.savefig("%s_%d.pdf"%(observations, N))
        # plt.show()

        # What do I need to save :



        # predict at datapoints:
        mus_data, vars_data = m.q.predict_fs(X, full_output_cov=False)
        N_grid = 1000
        X_grid = np.hstack([np.linspace(r[0], r[1], N_grid).reshape(-1, 1) for r in xranges])
        mus_grid, vars_grid = m.q.predict_fs(X_grid, full_output_cov=False)
        mup_data, varp_data = m.predict_predictor(X)

        F

        results = {
        'X':X,
        'Y':Y,
        'F':F,
        'rho':rho,
        'logf':logf,
        'mus_data':mus_data.numpy(),
        'vars_data':vars_data.numpy(),
        'X_grid' :X_grid,
        'mus_grid':mus_grid.numpy(),
        'vars_grid':vars_grid.numpy(),
        'mup_data':mup_data.numpy(),
        'varp_data':varp_data.numpy(),
        }

        for k, v in results.items():
            if isinstance(v, list):
                pass
            else:
               print(k, v.shape)

        mod = 'C' if coupled else 'MF'
        filename = 'exp_%s_%d.p' % (mod, N)
        pickle.dump(results, open(filename, 'wb'))



