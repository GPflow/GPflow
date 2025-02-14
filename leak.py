import gc

import keras.backend
import numpy as np
import psutil
import tensorflow as tf

import gpflow

X = np.array([[0.865], [0.666], [0.804], [0.771], [0.147], [0.866], [0.007], [0.026], [0.171], [0.889], [0.243], [0.028]])
Y = np.array([[1.57], [3.48], [3.12], [3.91], [3.07], [1.35], [3.80], [3.82], [3.49], [1.30], [4.00], [3.82]])
kernel = gpflow.kernels.SquaredExponential()
model = gpflow.models.GPR((X, Y), kernel=kernel)

opt = gpflow.optimizers.Scipy(compile_cache_size=0)

for i in range(400):
    # A. garbage collect to remove noise
    keras.backend.clear_session(); gc.collect()

    # B. define closure
    # closure = model.training_loss_closure(compile=True)
    # closure = model.training_loss
    # closure = model.log_posterior_density
    # closure = model.log_marginal_likelihood
    closure = lambda: kernel(X)

    # C. compile closure (and maybe more)
    # opt.minimize(closure, model.trainable_variables, compile=True)
    tf.function(closure)()

    print(f"[{i+1}]", psutil.virtual_memory())

