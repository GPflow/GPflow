import numpy as np
import tensorflow as tf
import GPflow

from GPflow._settings import settings
float_type = settings.dtypes.float_type
np_float_type = np.float32 if float_type is tf.float32 else np.float64

from itertools import chain
import pdb

import matplotlib.pyplot as plt

def embedsubmat(A, shape):
    '''Embeds A in a zero matrix with shape `shape` in the lower-right corner.'''
    sizediff = tf.expand_dims(shape - tf.shape(A), 1)
    paddings = tf.pad(sizediff, [[0, 0], [0, 1]])
    return tf.pad(A, paddings)


class MultiKernel(GPflow.kernels.Kern):
    '''this abstract kernel assumes input X where the first column is a series of integer indices and the
    remaining dimensions are unconstrained. Multikernels are designed to handle outputs from different
    Gaussian processes, specifically in the case where they are not independent and where they can be
    observed independently. This abstract class implements the functionality necessary to split
    the observations into different cases and reorder the final kernel matrix appropriately.'''
    def __init__(self, input_dim, groups, active_dims=None):
        GPflow.kernels.Kern.__init__(self, input_dim, active_dims)
        assert(input_dim > 1)
        self.groups = groups

    def multikernel(self, Xparts, X2parts):
        '''this method computes the kernel matrix for a sorted and partitioned dataset'''
        return NotImplementedError

    def multidiag(self, Xparts, X2parts):
        '''this method computes the diagonal of the kernel matrix
        for a sorted and partitioned dataset'''
        return NotImplementedError

    def K(self, X, X2=None):
        X, X2 = self._slice(X, X2)
        Xindex = tf.cast(X[:, 0], tf.int32) #find group indices
        Xparts, Xsplitn, Xreturn = self.splitback(X[:,1:], Xindex)
        #noise = 0.0

        if X2 is None:
            X2, X2parts, X2return, X2splitn = (X, Xparts, Xreturn, Xsplitn)
            #noise = tf.diag(self.multinoise(Xindex))
        else:
            X2index = tf.cast(X2[:, 0], tf.int32)
            X2parts, X2splitn, X2return = self.splitback(X2[:,1:], X2index)

        #construct kernel matrix for index-sorted data (stacked Xparts)
        Ksort = self.multikernel(Xparts, X2parts)

        #split matrix into chunks, then stitch them together in correct order
        Ktmp = self.reconstruct(Ksort, Xsplitn, Xreturn)
        KT = self.reconstruct(tf.transpose(Ktmp), X2splitn, X2return)
        return tf.transpose(KT)

    def Kdiag(self, X):
        X, _ = self._slice(X, None)
        F = tf.cast(X[:, 0], tf.int32) #find recursion level indices
        Xparts, Xsplitn, Freturn = self.splitback(X[:,1:], F)
        Kd = self.multidiag(Xparts)
        return self.reconstruct(Kd, Xsplitn, Freturn)
        #return self.reconstruct(Kd, Xsplitn, Freturn) + self.multinoise(F)

    #def multinoise(self, indices):
    #    '''distribute block-dependent noise'''
    #    return tf.gather(self.noises, indices)

    def splitback(self, data, indices):
        '''applies dynamic_partioning and calculates necessary statistics for
        the inverse mapping.'''
        parts = tf.dynamic_partition(data, indices, self.groups) #split data based on indices

        #to be able to invert the splitting, we need:
        splitnum = tf.stack([tf.shape(x)[0] for x in parts]) #the size of each data split
        goback = tf.dynamic_partition(tf.range(tf.shape(data)[0]), indices, self.groups) #indices to invert dynamic_part
        return (parts, splitnum, goback)

    def reconstruct(self, K, splitnum, goback):
        '''uses quantities from splitback to invert a dynamic_partition'''
        tmp = tf.split(K, splitnum, axis=0) #split
        return tf.dynamic_stitch(goback, tmp) #stitch

class HeteroscedasticWhite(GPflow.kernels.Kern):
    '''Apply different noise to observations with different index.'''
    def __init__(self, input_dim, groups, active_dims=None):
        GPflow.kernels.Kern.__init__(self, input_dim, active_dims=None)
        self.groups = groups
        self.noises = GPflow.param.Param(1e-3*np.ones(self.groups), GPflow.transforms.positive)

    def K(self, X, X2=None, presliced=False):
        '''Compute noise kernel. If `X2` is not None, returns 0.0. If `X2` is
        `None` the function offloads the workload to `Kdiag` as the kernel is
        diagonal.
        '''
        if not presliced:
            X, X2 = self._slice(X, X2)
        if X2 is None:
            return 0.0
        return tf.diag(self.Kdiag(X, presliced=True))

    def Kdiag(self, X, presliced=False):
        '''Computes the diagonal of the kernel matrix. Reads off group indices by
        by casting the single input to integers. It then distributes the noise levels
        according to group index.

        Parameters:
        - `X`: Tensor. Assumed that it can be cast to integers in the range [0,self.groups)
        '''
        if not presliced:
            X, _ = self._slice(X, None)
        indices = tf.cast(X[:, 0], tf.int32)
        return tf.gather(self.noises, indices)

class RecursiveKernel(MultiKernel):
    '''This kernel assumes that inputs with index 0 are generated from a single GP,
    that inputs with index 1 are generated from that GP plus another independent GP,
    and so on, recursively. This leads to N covariant generative models, and a covariance matrix
    where submatrices corresponding to a particular index equals the sum of the covariance kernels of all GPs
    with lower or equal index.'''

    def subKdiag(self, index, X):
        return NotImplementedError

    def subK(self, index, X, X2=None):
        return NotImplementedError

    def multikernel(self, Xparts, X2parts):
        '''this method computes the kernel matrix for a sorted and partitioned dataset'''
        #build initial empty tensors
        K = tf.zeros((0,0), dtype=tf.float64)
        D = tf.zeros((0,self.input_dim-1), dtype=tf.float64)
        D2 = tf.zeros((0,self.input_dim-1), dtype=tf.float64)

        #zip dataset
        XandX2 = zip(Xparts, X2parts)
        #reverse for algorithmic convenience
        XandX2 = list(reversed(list(XandX2)))

        #kernel shape if all data chunks up to index i are joined.
        XandX2shapes = tf.cumsum(tf.stack([(tf.shape(d1)[0],tf.shape(d2)[0]) for d1,d2 in XandX2]), axis=0)

        for index, (data1, data2) in enumerate(XandX2):
            #add incoming data
            D = tf.concat([data1, D], 0)
            D2 = tf.concat([data2, D2], 0)

            #look up shape of the new kernel
            nshape = XandX2shapes[index,:]

            #calculate kernel for current level
            k_new = self.subK(index, D, D2)

            # grow kernel matrix
            K = k_new + embedsubmat(K, nshape)
        return K

    def multidiag(self, Xparts):
        '''this method computes the diagonal of the kernel matrix
        for a sorted and partitioned dataset'''
        datarev = list(reversed(list(Xparts)))
        Xshapes = tf.cumsum(tf.stack([tf.shape(d)[0] for d in datarev]), axis=0)
        Kd = tf.zeros((0), dtype=tf.float64)
        D = tf.zeros((0, self.input_dim-1), dtype=tf.float64)
        for index, data in enumerate(datarev):
            D = tf.concat([data, D], 0)
            sizediff = Xshapes[index]-tf.shape(Kd)[0]
            Kd = tf.pad(Kd,[[sizediff,0]])  +  self.subKdiag(index, D)
        return Kd

class BlockKernel(MultiKernel):
    '''this kernel applies constructs each block of the kernel matrix separately.'''
    def __init__(self, input_dim, groups, active_dims=None):
        MultiKernel.__init__(self, input_dim, groups, active_dims)

    def multikernel(self, Xparts, X2parts):
        '''this method computes the kernel matrix for a sorted and partitioned dataset'''

        #build initial empty tensors
        K = tf.zeros((0,0), dtype=tf.float64)
        D = tf.zeros((0,self.input_dim-1), dtype=tf.float64)
        D2 = tf.zeros((0,self.input_dim-1), dtype=tf.float64)

        #loop over all cases to iteratively construct block matrix
        rows = []
        for i in range(self.groups):
            row_i = []
            for j in range(self.groups):
                row_i.append(self.subK((i, j), Xparts[i], X2parts[j]))
            rows.append(tf.concat(row_i, 1))
        return tf.concat(rows, 0)

    def multidiag(self, Xparts):
        Kd = tf.zeros((0), dtype=tf.float64)
        D = tf.zeros((0, self.input_dim-1), dtype=tf.float64)
        subdiags = []
        for index, data in enumerate(Xparts):
            subdiags.append(self.subKdiag(index, Xparts[index]))
        return tf.concat(subdiags, 0)


class KernelList(GPflow.param.Parameterized):
    '''This class enables you to store multiple kernels inside your model and
    have all automatic parameter discovery methods function as designed.
    '''
    def __init__(self, kerns, nested = False):
        GPflow.param.Parameterized.__init__(self)
        self.keydict = {}
        if nested:
            for i, row in enumerate(kerns):
                for j, k in enumerate(row):
                    key = (i, j)
                    name =  'kernel_{}{}'.format(*key)
                    setattr(self, name, k)
                    self.keydict[key] = name
        else:
            for i, k in enumerate(kerns):
                key = i
                name =  'kernel_{}'.format(key)
                setattr(self, name, k)
                self.keydict[key] = name

    def __getitem__(self, key):
        return getattr(self, self.keydict[key])

class KernelKeeper:
    '''This mix-in class adds a list of kernels to a class'''
    def __init__(self, kerns):
        try:
            self.kerns = KernelList(kerns, nested = True)
        except TypeError:
            self.kerns = KernelList(kerns, nested = False)


class BlockLookupKernel(KernelKeeper, BlockKernel):
    '''This kernel keeps a matrix of kernels that it consults to fill out blocks.
    Note that this kernel is only valid for very particular choices of kernels.'''
    def __init__(self, input_dim, kerns, active_dims=None):
        BlockKernel.__init__(self, input_dim, len(kerns), active_dims)
        KernelKeeper.__init__(self, kerns)

    def subK(self, index, X, Y = None):
        i, j = index
        if Y is None:
            Y = X
        return self.kerns[i, j].K(X, Y, presliced = True)

    def subKdiag(self, index, X):
        i = index
        if Y is None:
            Y = X
        return self.kerns[i].Kdiag(X, presliced = True)

class RecursiveLookupKernel(KernelKeeper, RecursiveKernel):
    '''This kernel does co-kriging using a list of N independent kernels'''
    def __init__(self, input_dim, kerns, active_dims=None):
        MultiKernel.__init__(self, input_dim, len(kerns), active_dims)
        KernelKeeper.__init__(self, kerns)

    def subKdiag(self, index, X):
        return self.kerns[index].Kdiag(X, presliced=True)

    def subK(self, index, X, X2=None):
        return self.kerns[index].K(X, X2, presliced=True)


ind = np.random.randint(0,3,(20,1))
val = ind.astype(np.float64) + ind*0.5*np.random.randn(20,1)
X = np.column_stack([ind, val])

#block kernel won't work for arbitrary kernels, but can be defined as follows:
#BK = BlockLookupKernel(2, [[GPflow.kernels.RBF(1) for _ in range(3)] for _ in range(3)])

#the Recursive kernel is always valid:
RK = RecursiveLookupKernel(2, [GPflow.kernels.RBF(1) for _ in range(3)])

m = GPflow.gpr.GPR(X, 0.1*val**2.*np.exp(-val**2.), RK)
m.optimize()
