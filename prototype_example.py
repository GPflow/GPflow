import sys
import csv
import numpy as np
import gpflow
import tensorflow as tf

Xtrain = np.loadtxt('notebooks/data/banana_X_train', delimiter=',')
Ytrain = np.loadtxt('notebooks/data/banana_Y_train', delimiter=',').reshape(-1, 1)

idx = np.random.choice(range(Xtrain.shape[0]), size=3, replace=False)
feature = Xtrain[idx, ...]

# 1. `input_dims` are not required anymore.
kernel = gpflow.kernels.RBF()

# 2. Assigned value (10.0) here is constrained.
kernel.lengthscales <<= 10.0
kernel.variance.trainable = False
likelihood = gpflow.likelihoods.Bernoulli()

# 3. Constrained vs unconstrained values.
print(f"Unconstrained parameter value of `kernel.lengthscales` = {kernel.lengthscales}")
print(f"Constrained parameter value of `kernel.lengthscales` = {kernel.lengthscales()}")

# 4. X's and Y's are no longer part of the model.
m = gpflow.models.SVGP(kernel=kernel, feature=feature, likelihood=likelihood)

X, Y = tf.convert_to_tensor(Xtrain), tf.convert_to_tensor(Ytrain)
def loss_cb():
    return m.neg_log_marginal_likelihood(X, Y)

# 5. There is no more gpflow optimizers.
adam = tf.train.AdamOptimizer(0.0001)

# 6. Keras-like model fitting
gpflow.optimize(loss_cb, adam, m.trainable_variables, 10)