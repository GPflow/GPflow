import numpy as np
import tensorflow as tf
import GPflow
from GPflow.param import Param
from GPflow import transforms
import pdb

import matplotlib.pyplot as plt

from functools import partial

from multikernel import BlockKernel

from GPflow._settings import settings
float_type = settings.dtypes.float_type


def find_partitions(collection):
    '''Take an iterable `collection` and return a generator which produces
    all unique partitions as lists-of-lists with every object being a member of
    one set per partition.
    alexis @ http://stackoverflow.com/questions/19368375/set-partitions-in-python
    '''
    if len(collection) == 1:
        yield [ collection ]
        return

    first = collection[0]
    for smaller in find_partitions(collection[1:]):
        # insert `first` in each of the subpartition's subsets
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[ first ] + subset]  + smaller[n+1:]
        # put `first` in its own subset
        yield [ [ first ] ] + smaller

class StationaryGradKernel(BlockKernel):
    ''' Abstract kernel class which allows arbitrary derivative observations for
    any differentiable stationary kernel. Computes each kernel matrix as a
    composition of block matrices and automatically generates appropriate kernel
    functions if given derivatives of the kernel matrix wrt squared distance.

    To extend, inherit and replace `derivative_base` attribute with list of
    the 0th, 1st, and 2nd derivative of desired kernel k(tau) with respect to squared
    distance tau. Defaults to RBF/squared exponential.
    '''
    def __init__(self, input_dim, active_dims = None):
        groups = input_dim
        BlockKernel.__init__(self, input_dim, groups, active_dims)
        self.variance = GPflow.param.Param(0.1, transforms.positive)
        self.lengthscales = GPflow.param.Param(1.*np.ones(input_dim-1), transforms.positive)

        self.derivative_base = [lambda tau: self.variance*tf.exp(-0.5*tau), lambda tau: -0.5*self.variance*tf.exp(-0.5*tau), lambda tau: 0.25*self.variance*tf.exp(-0.5*tau)]
        self.kerns = [[self._kernel_factory(i,j) for j in range(groups)] for i in range(groups)]

    def subK(self, index, X, X2 = None):
        i, j = index
        if X2 is None:
            X2 = X
        return self.kerns[i][j](X, X2)

    def subKdiag(self, index, X):
        K  = self.subK((index, index), X, X)
        return tf.diag_part(K)

    def square_dist(self, X, X2):
        '''Function `squared_dist` identical to method in `Stationary`.'''
        X = X / self.lengthscales
        Xs = tf.reduce_sum(tf.square(X), 1)
        if X2 is None:
            return -2 * tf.matmul(X, tf.transpose(X)) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
        else:
            X2 = X2 / self.lengthscales
            X2s = tf.reduce_sum(tf.square(X2), 1)
            return -2 * tf.matmul(X, tf.transpose(X2)) + \
                   tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))

    def square_dist_d(self, X, X2, diff_x=(), diff_y=()):
        '''Take derivative of square_dist(x,y) wrt dimensions `diff_x` of first
        argument and `diff_y` of second argument.

        Parameters:
        - `X` and `X2`: arguments for `square_dist`.
        - `diff_x`: Tuple of integers. Dimensions to differentiate with respect
            to in the first argument of `square_dist(x,y)`.
        - `diff_y`: Tuple of integers. Dimensions to differentiate with respect
            to in the second argument of `square_dist(x,y)`.

        Return:
        - Appropriate derivative as a Tensorflow Tensor.
        '''
        x_order = len(diff_x)
        y_order = len(diff_y)
        diff_order = x_order + y_order
        dx = diff_x + diff_y

        Zero =  tf.zeros((tf.shape(X)[0], tf.shape(X)[0] if X2 is None else tf.shape(X2)[0]), dtype=float_type)
        One = tf.ones((tf.shape(X)[0], tf.shape(X)[0] if X2 is None else tf.shape(X2)[0]), dtype=float_type)

        if diff_order > 2:
            #higher derivatives vanish
            return Zero
        elif diff_order == 0:
            #if we don't take the derivative, return the function
            return self.square_dist(X, X2)
        elif diff_order == 1:

            X = X / self.lengthscales ** 2
            if X2 is None:
                X2 = X
            else:
                X2 = X2 / self.lengthscales ** 2
            delta = tf.expand_dims(X[:,dx[0]], 1) - tf.expand_dims(X2[:,dx[0]], 0)
            #sign-flip if y derivative
            if x_order > 0:
                return 2.*delta
            elif y_order > 0:
                return -2.*delta

        elif diff_order == 2:
            #only equal-index dimension pairs are non-zero
            i, j = dx
            if i == j:
                k = One*(2./self.lengthscales[i]**2.)
                if x_order == y_order:
                    return -k
                else:
                    return k
            else:
                return Zero
        raise RuntimeError

    def ind2dx(self, index):
        '''Convert a linear index into a tuple of integers to differentiate with respect to.'''
        return () if index == 0 else (index-1,)

    def kernel_derivative(self, X, X2, diff_x=(), diff_y=()):
        '''Calculates derivative of stationary kernel by decomposing into
        derivatives of the kernel wrt the distance and derivatives of the distance
        wrt the inputs. Procedurally generates each term of the derivative by
        iterating over all possible products of mixed derivatives.

        Parameters:
        - `X` and `X2`: Numpy/TF array. Inputs over which the kernel is evaluated.
        - `diff_x` and `diff_y`: Tuples of

        Return:
        - `result` -
        '''
        dist_d = partial(self.square_dist_d, X=X, X2=X2)
        dx = [(x,'x') for x in diff_x] + [(y,'y') for y in diff_y]
        x_order = len(diff_x)
        y_order = len(diff_y)
        diff_order = len(dx)
        if diff_order == 0:
            return self.derivative_base[0](self.square_dist(X, X2))
        dx_combinations = find_partitions(dx)
        order_factor = [self.derivative_base[order](self.square_dist(X, X2))
                        for order in range(diff_order+1)]
        terms = []
        for partition in dx_combinations:
            order = len(partition)
            derivatives = [dist_d(diff_x=tuple(i for i, key in elem if key is 'x'),
                                  diff_y=tuple(j for j, key in elem if key is 'y'))
                           for elem in partition]
            terms.append(order_factor[order]*tf.foldl(tf.multiply, derivatives))
        result = tf.foldl(tf.add, terms)
        return result

    def _kernel_factory(self, i, j):
        '''Return a function that calculates proper sub-kernel'''
        diff_x = self.ind2dx(i)
        diff_y = self.ind2dx(j)
        def f(X, X2):
            return self.kernel_derivative(X, X2, diff_x=diff_x, diff_y=diff_y)
        return f

class StationaryHessianKernel(StationaryGradKernel):
    '''Generalizes StationaryGradKernel to second deriatives'''
    def __init__(self, input_dim, active_dims = None):
        ndim = input_dim-1
        groups = 1 + ndim + ((ndim)*(ndim+1))//2
        BlockKernel.__init__(self, input_dim, groups, active_dims)
        self.variance = GPflow.param.Param(0.1, transforms.positive)
        self.lengthscales = GPflow.param.Param(1.*np.ones(input_dim-1), transforms.positive)
        self.tril_indices = np.column_stack(np.tril_indices(ndim))
        self.derivative_base = [lambda tau: self.variance*tf.exp(-0.5*tau), lambda tau: -0.5*self.variance*tf.exp(-0.5*tau), lambda tau: 0.25*self.variance*tf.exp(-0.5*tau),
        lambda tau: -0.125*self.variance*tf.exp(-0.5*tau), lambda tau: 0.0625*self.variance*tf.exp(-0.5*tau)]
        #self.derivative_base = [lambda tau: self.variance*(-0.5)**n*tf.exp(-0.5*tau) for n in range(5)]
        self.kerns = [[self._kernel_factory(i,j) for j in range(groups)] for i in range(groups)]


    def ind2dx(self, i):
        if i == 0:
            return ()
        elif i<self.input_dim:
            return (i-1,)
        else:
            return tuple(self.tril_indices[i-self.input_dim,:])


d = 1
groups = 3
Ns = [10,5,0]

ind = np.vstack([n*np.ones((Ns[n],1)) for n in range(3)])
N = np.sum(Ns)
#np.random.shuffle(ind)
val = np.vstack([np.linspace(0,1,n).reshape((-1,1)) for n in Ns])
X = np.concatenate([ind, val],1)
f = lambda x: np.exp(-x)
fp = lambda x: -np.exp(-x)
fpp = lambda x: np.exp(-x)
y = np.vstack([fun(val[ind == i]).reshape((-1,1)) for fun, i in zip([f,fp,fpp], range(groups))])


k = StationaryHessianKernel(d+1)
kg = StationaryGradKernel(d+1)
m = GPflow.gpr.GPR(X, y, k)
mg = GPflow.gpr.GPR(X, y, kg)
m.likelihood.variance = 0.1
mg.likelihood.variance = 0.1

K = k.compute_K(X, X)
Kg = kg.compute_K(X, X)
#m.optimize()
#mg.optimize()
res = 100
x = np.column_stack([np.linspace(-1,1, res),np.zeros(res)])
predlevel = lambda i: m.predict_f(np.column_stack([i*np.ones(res),x]))
mu, var = predlevel(0)
mug, varg = predlevel(1)
mug2, varg = predlevel(2)

plt.plot(x[:,0],mu,label='Function')
plt.plot(x[:,0],mug,label='Derivative')
plt.plot(x[:,0],mug2,label='Derivative2')

plt.hlines(0, -1, 1)
plt.legend()
