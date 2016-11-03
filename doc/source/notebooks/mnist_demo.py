import numpy as np
import GPflow
import cPickle as pickle
import time
import tensorflow as tf
import urllib2
import os.path
from tensorflow.examples.tutorials.mnist import input_data

data_directory = 'data/'
mnist_file_name = 'mnist_pickle'

def getMnistData():
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
	X_train = np.vstack( [mnist.train.images,mnist.validation.images] )
	Y_train = np.atleast_2d(np.hstack( [mnist.train.labels,mnist.validation.labels] )).T
	X_test = mnist.test.images
	Y_test = mnist.test.labels
	return X_train, Y_train, X_test, Y_test

np.random.seed(0)
ndata = None # none for all data
num_inducing = 500

vb_max_iters = [20000,30000,40000,50000,60000]
beta = 0.5
step_rates = [ 1e-3 * (beta ** elem ) for elem in range(5) ]

nClasses = 10
vb_batchsize = 1000
thin = 2

X_train, Y_train, X_test, Y_test = getMnistData()

#scale data
X_train = X_train/255.0
X_train = X_train*2. - 1.
X_test = X_test/255.0
X_test = X_test*2. - 1.

#randomize order
rng = np.random.RandomState(0)
i = rng.permutation(X_train.shape[0])
i = i[:ndata]
X_train, Y_train = X_train[i,:], Y_train[i,:]

from scipy.cluster.vq import kmeans
skip = 20
initZ, _ = kmeans(X_train[::skip,:], num_inducing)

def getKernel():
    k = GPflow.kernels.RBF(X_train.shape[1], ARD=False) + GPflow.kernels.White(1, 1e-3)
    return k
    
m_vb = GPflow.svgp.SVGP(X=X_train, Y=Y_train.astype(np.int32), kern=getKernel(), likelihood=GPflow.likelihoods.MultiClass(nClasses), num_latent=nClasses, Z=initZ.copy(), minibatch_size=vb_batchsize, whiten=False)
m_vb.q_mu = m_vb.q_mu.value+np.random.randn(*m_vb.q_mu.value.shape)*0.5 #Add jitter to initial function values to move away from local extremum of objective.

m_vb.likelihood.invlink.epsilon = 1e-3
m_vb.likelihood.fixed=True

#Takes a long time to run.        
for repeatIndex in range(len(vb_max_iters)):
	print "repeatIndex ", repeatIndex
	start_time = time.time()
	m_vb.optimize( tf.train.AdadeltaOptimizer(learning_rate=step_rates[repeatIndex], rho=0.9, epsilon=1e-4, use_locking=True) , maxiter=vb_max_iters[repeatIndex] )
	mu, _ = m_vb.predict_y(X_test)
	percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
	new_time = time.time()
	print "percent ", percent
	print "cycle_diff ", new_time - start_time
	start_time = new_time

#achieves 97.94% error.
