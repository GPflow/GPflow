import numpy as np
import tensorflow as tf
import gpflow as gpf

N = 100
D = 1

Fmu = np.zeros((N, D)) * 1.
Fvar= np.ones((N, D)) * 1.
Y = np.ones((N, D)) * 1.

lik = gpf.likelihoods.Bernoulli(invlink=tf.sigmoid)
print(lik.variational_expectations(Fmu, Fvar, Y).numpy())
