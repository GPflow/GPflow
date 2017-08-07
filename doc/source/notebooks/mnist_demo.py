from __future__ import print_function
import numpy as np
import GPflow
import time
import tensorflow as tf
import os.path
from GPflow import settings
from tensorflow.examples.tutorials.mnist import input_data

def getMnistData():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False, seed=1)
	X_train = np.vstack( [mnist.train.images,mnist.validation.images] )
	Y_train = np.atleast_2d(np.hstack( [mnist.train.labels,mnist.validation.labels] )).T
	X_test = mnist.test.images
	Y_test = mnist.test.labels
	#scale data
	X_train = X_train*2. - 1.
	X_test = X_test*2. - 1.
	return X_train, Y_train, X_test, Y_test

num_inducing = 500

vb_frozen_iters = 8000
vb_max_iters = 200000
vb_batchsize = 1000
step_rates = 1e-1, 5e-2

nClasses = 10

X_train, Y_train, X_test, Y_test = getMnistData()

np.random.seed(0)
from scipy.cluster.vq import kmeans2 as kmeans
skip = 20
initZ, _ = kmeans(X_train[::skip,:], num_inducing, minit='points')

def getKernel():
    k = GPflow.kernels.RBF(X_train.shape[1], ARD=False) + GPflow.kernels.White(1, 1e-3)
    return k
    
rng = np.random.RandomState(1)
m_vb = GPflow.svgp.SVGP(X=X_train, Y=Y_train.astype(np.int32), kern=getKernel(), likelihood=GPflow.likelihoods.MultiClass(nClasses), num_latent=nClasses, Z=initZ.copy(), minibatch_size=vb_batchsize, whiten=False)
m_vb.q_mu = m_vb.q_mu.value+rng.randn(*m_vb.q_mu.value.shape)*0.5 #Add jitter to initial function values to move away from local extremum of objective.

m_vb.likelihood.invlink.epsilon = 1e-3
m_vb.likelihood.fixed=True

m_vb.kern.fixed=True
m_vb.Z.fixed=True
start_time = time.time()
m_vb.optimize( tf.train.AdadeltaOptimizer(learning_rate=step_rates[0], rho=0.9, epsilon=1e-4, use_locking=True) , maxiter=vb_frozen_iters )#
mu, _ = m_vb.predict_y(X_test)
percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
print("percent ", percent)
new_time = time.time()
print("cycle_diff ", new_time - start_time)
start_time = new_time
m_vb.kern.fixed=False
m_vb.Z.fixed=False
m_vb.optimize( tf.train.AdadeltaOptimizer(learning_rate=step_rates[1], rho=0.9, epsilon=1e-4, use_locking=True) , maxiter=vb_max_iters )#
mu, _ = m_vb.predict_y(X_test)
percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
print("percent ", percent)