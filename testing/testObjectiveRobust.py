import GPflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import GPy
import time
import tensorflow as tf

X = np.array(pd.read_csv('https://gist.githubusercontent.com/NMRobert/8ea8b621fa12042b1dd78dfb9ca31371/raw/15a7432c04a1c6151737893b9096144619ec1ff0/X.csv', sep=',',header=None))
y = np.array(pd.read_csv('https://gist.githubusercontent.com/NMRobert/8ea8b621fa12042b1dd78dfb9ca31371/raw/15a7432c04a1c6151737893b9096144619ec1ff0/y.csv', sep=',', header=None))
dim = np.shape(X)[1]
print 'Downloaded X,y from gist.'

def GoodNumber(a):
    assert np.any(np.logical_or(np.isnan(a),np.isinf(a))) == False, 'Should not have any nans==> ' + str(a[:3])

def GPyRegression():
    k1 = GPy.kern.RBF(input_dim=dim, ARD=False)
    k2 = GPy.kern.Linear(input_dim=7, active_dims=[10, 11, 12, 13, 14, 15, 16])
    k3 = GPy.kern.Matern32(input_dim=3, active_dims=[0, 7, 8], ARD=True)
    k4 = GPy.kern.StdPeriodic(input_dim=3, active_dims=[0, 1, 2])
    k = (k1 + k4 + k2 * k3)

    m = GPy.models.GPRegression(X, y, k)
    m.optimize()
    mean, v = m.predict(X)
    GoodNumber(mean)
    GoodNumber(v)
#    plot(range(len(X)), y, mean)
    return m


def GPflowRegression(o='L-BFGS-B',max_iters=30):
    tf.reset_default_graph()
    k1 = GPflow.kernels.RBF(input_dim=dim, ARD=False)
    k2 = GPflow.kernels.Linear(input_dim=7, active_dims=[10, 11, 12, 13, 14, 15, 16])
    k3 = GPflow.kernels.Matern32(input_dim=3, active_dims=[0,7,8])
    k4 =GPflow.kernels.PeriodicKernel(input_dim=3, active_dims=[0, 1, 2])
    k = (k1 + k4 + k2 * k3)

    m = GPflow.gpr.GPR(X, y, k)
    m.optimize(o,max_iters=max_iters,options={'disp':True})
    GoodNumber(m.get_free_state())
    print(m)
    mean, v = m.predict_y(X)
    GoodNumber(mean)
    GoodNumber(v)
#    plot(range(len(X)), y, mean)
    return m


def plot(X,y, mean):
    plt.plot(range(len(X)), y, color='red', linestyle='dotted', linewidth=2)
    plt.plot(range(len(X)), mean, color='blue', linewidth=2, linestyle='dashed')
    plt.show()

#print '-----------------BFGS-------------------------' # this fails with prediction giving nans
#t = time.time()
#m1 = GPflowRegression('BFGS') 
#print 'GPflow took %g secs.'%(time.time()-t)


print '-----------------L-BFGS-B-------------------------'
t = time.time()
m1 = GPflowRegression('L-BFGS-B') 
print 'GPflow took %g secs.'%(time.time()-t)

print '-----------------CG-------------------------'
t = time.time()
m2 = GPflowRegression('CG') 
print 'GPflow took %g secs.'%(time.time()-t)

print '-----------------TNC-------------------------' # this does not hit the exception
t = time.time()
m3 = GPflowRegression('TNC') 
print 'GPflow took %g secs.'%(time.time()-t)

#print '-----------------SLSQP-------------------------' # this gets stuck
#t = time.time()
#m4 = GPflowRegression('SLSQP') 
#print 'GPflow took %g secs.'%(time.time()-t)


print '-----------------tf AdamOptimizer-------------------------'
t = time.time()
o = tf.train.AdamOptimizer()
m5 = GPflowRegression(o) 
print 'GPflow took %g secs.'%(time.time()-t)

#t = time.time()
#mgpy = GPyRegression() 
#print mgpy
#print 'GPy took %g secs.'%(time.time()-t)

