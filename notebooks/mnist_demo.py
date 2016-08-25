#This version of the MNIST experiment is for profiling purposes only and is not yet ready for general use.
#I plan to submit a pull request to GPflow for it when I am satisfied it is ready.
#Alexander G. de G. Matthews

#Requires mnist_pickle which I won't be checking in.

from matplotlib import pylab as plt
import numpy as np
import GPflow
#import climin
import sys
import cPickle as pickle
import time
import tensorflow as tf
from IPython import embed

np.random.seed(0)
reduced_dim = 45
ndata = None # none for all data
num_inducing = 500
ahmc_num_steps = 50
ahmc_s_per_step = 20
hmc_num_samples = 300
vb_frozen_iters = 8000
nClasses = 10

#Settings for profiler
nFrozenRepeats = 1
vb_max_iters = [6]
step_rates = [0.1]

vb_batchsize = 1000
thin = 2

data = pickle.load(open('mnist_pickle','r'))
X_train, Y_train, X_test, Y_test = data['Xtrain'], data['Ytrain'], data['Xtest'], data['Ytest']
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

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


#build vb model
def getKernel():
    k = GPflow.kernels.RBF(X_train.shape[1], ARD=False) + GPflow.kernels.White(1, 1e-3)
    return k
    
class cb():
    def __init__(self, model, holdout_interval=10 ):
        self.holdout_interval = holdout_interval
        self.percentages = []
        self.n_iters = []
        self.counter = 0
        self.local_model = model 
        
    def __call__(self, info):
        if (self.counter%self.holdout_interval)==0:
            self.n_iters.append( self.counter )
            mu, _ = self.local_model.predict_y(X_test)
            percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
            #self.hold_out_likelihood.append( dm )
            self.percentages.append( percent )
            print "percent ", percent, " \n"
        self.counter+=1


m_vb = GPflow.svgp.SVGP(X=X_train, Y=Y_train.astype(np.int64), kern=getKernel(), likelihood=GPflow.likelihoods.MultiClass(nClasses), num_latent=nClasses, Z=initZ.copy(), minibatch_size=vb_batchsize, whiten=False)

m_vb.likelihood.invlink.epsilon = 1e-3
m_vb.likelihood.fixed=True

for repeatIndex in range(len(vb_max_iters)):
	print "repeatIndex ", repeatIndex
	start_time = time.time()
	if repeatIndex<nFrozenRepeats:
		m_vb.kern.fixed=True
		m_vb.Z.fixed=True
	else:
		m_vb.kern.fixed=False
		m_vb.Z.fixed=False
	m_vb.optimize( tf.train.AdadeltaOptimizer(learning_rate=step_rates[repeatIndex], rho=0.9, epsilon=1e-4, use_locking=True) , maxiter=vb_max_iters[repeatIndex] )
	mu, _ = m_vb.predict_y(X_test)
	percent = np.mean(np.argmax(mu,1)==Y_test.flatten())
	new_time = time.time()
	print "percent ", percent
	print "cycle_diff ", new_time - start_time
	start_time = new_time

embed()
