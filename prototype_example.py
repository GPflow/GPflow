import sys
import csv
import numpy as np
import gpflow
import tensorflow as tf

Xtrain = np.loadtxt('notebooks/data/banana_X_train', delimiter=',')
Ytrain = np.loadtxt('notebooks/data/banana_Y_train', delimiter=',').reshape(-1, 1)

idx = np.random.choice(range(Xtrain.shape[0]), size=3, replace=False)
feature = Xtrain[idx, ...]

kernel = gpflow.kernels.RBF()
kernel.lengthscales <<= 10.0
kernel.variance.trainable = False
likelihood = gpflow.likelihoods.Bernoulli()
m = gpflow.models.SVGP(kernel=kernel, feature=feature, likelihood=likelihood)

X, Y = tf.convert_to_tensor(Xtrain), tf.convert_to_tensor(Ytrain)
def loss_cb():
    return m.neg_log_marginal_likelihood(X, Y)

adam = tf.train.AdamOptimizer(0.0001)
gpflow.optimize(loss_cb, adam, m.trainable_variables, 10)
print(m.variables)