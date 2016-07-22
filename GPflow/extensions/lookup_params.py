# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ..param import Param, Parameterized
from ..data_holders import DictData
from GPflow import transforms
            

class LookupParam(Parameterized):
    """
    Parameter object that accepts index.
    
    To instanciate the object, an initial parameter np.arrays should be 
    provided.
    
    >>> indexed_param = LookupParam(init_array)
    
    This object stores a GPflow.Param (indexed_param._table) with 
    first (several) dimension(s) for the indexed access.
    
    To access the GPflow.param, following methods are provided,
    
    with no tf_mode,
    >>> indexed_param(i)
    returns indexed_param._table.value[i]
    
    >>> indexed_param(i,j)
    returns indexed_param._table.value[i,j]
    
    >>> indexex_param.diag(i,j)
    returns if i==j indexed_param._table.value[i]
            otherwise np.zero
    
    with tf_mode, 
    1-dimensional-like integer tensor x (assuming x is length N) should be passed,
    (tensor is squeezed in the method, so tensor with shape [1,1,5,1] can be passed)

    >>> indexed_param(x)
    return is like [indexed_param._table[x_i] for i in range(len(x))]    

    >>> indexed_param(x,x)
    return is like [[indexed_param._table[x_j] for j in range(len(x))] for k in range(len(x))]
    
    >>> indexed_param.diag[x,x]
    return is like [[indexed_param._table[x_i] if x_i == x_j, 0. if x_i != x_j
                         for i in range(len(x))] for j in range(len(x))]]
    
    
    For the fix,
    >>> indexed_param.fixed = True 
    >>> indexed_param.fixed = False
    can be used as usual Parameterized object.
    
    To get the whole part of _table
    >>> current_table = indexed_param.table.value

    To replace _table,
    >>> indexed_param.table = np.array
    
    """
    
    def __init__(self, array, transform=transforms.Identity()):
        """
        :param np.array or list of np.array array: initial parameters.
        """
        Parameterized.__init__(self)
        self.table = Param(array, transform)

    
    def _begin_tf_mode(self):
        """
        Before beginning tf_mode, shape of table should be stored for appropriate
        transpose in __call__ method.
        
        The diag_generator is prepared for the use in diag() method.
        (This part should be improved.)
        """
        self.shape = self.table.value.shape
        super(LookupParam, self)._begin_tf_mode()
        
        # generating _diag_generator, that has the following elements
        # _diag_generator[i,j, k,..., ] = delta(i,j)
        # The shape of _diag_generator is _diag_generator.shape[0] = self.shape[0]
        # _diag_generator.shape[1:] = self.shape[0]
        self._diag_generator=tf.constant(np.identity(self.shape[0]))
        # Expand dims to make the dimension to match self.shape
        for i in range(len(self.shape)-1):
            self._diag_generator = tf.expand_dims(self._diag_generator, dim=-1)
        # multiplet to tile 
        mult = np.ones((len(self.shape)+1))
        mult[1:] = list(self.shape)
        mult[1] = 1
        self._diag_generator = tf.tile(self._diag_generator,\
            multiples=mult)



    def __call__(self, i, j=None):
        if j is None:
            if self._tf_mode:
                return tf.gather(self.table, i)
            else:
                return self.table.value[i]
        else:
            if self._tf_mode:
                # prepare permutation for swapping the axis0 and axis1
                perm=list(range(len(self.shape)))
                perm[0], perm[1] = 1, 0
                # gather according to index_i along axis0,
                # swap axis0 and axis1
                # gather according to index_j along axis0 (which is originally axis1),
                # then swap axis0 and axis1 again.
                return tf.transpose(tf.gather(\
                   tf.transpose(tf.gather(self.table, i), perm=perm)\
                                                    , j), perm=perm)
            else:
                return self.table.value[i,j]

    def diag(self, i, j):
        if self._tf_mode:
            # prepare multiples to generate a tensor that has the shape
            # [shape[0], shape[0], shape[1], ..., shape[N-1]]
            mult=np.ones(len(self.shape)+1)
            mult[0] = self.shape[0]
            # By Multiplying self._diag_generator, the resultant tensor is 
            # diag[i,i',j,k, ...,] = table[i,j,k,...,] delta(i-i')
            diag = self._diag_generator * tf.tile(tf.expand_dims(self.table, dim=0), multiples=mult)
            
            # prepare permutation for swapping the axis0 and axis1
            perm=list(range(len(self.shape)+1))
            perm[0], perm[1] = 1, 0
            # gather according to index_i along axis0,
            # swap axis0 and axis1
            # gather according to index_j along axis0 (which is originally axis1),
            # then swap axis0 and axis1 again.
            return tf.transpose(tf.gather(\
                       tf.transpose(tf.gather(diag, i), perm=perm)\
                                                  , j), perm=perm)
        else:
            if i == j:
                return self.table.value[i]
            else:
                return 0.


class LookupDictData(LookupParam):
    """
    DataHolder object that accepts index.
    The other functionalities are the same to LookupParam.
    """
    def __init__(self, array):
        """
        :param np.array or list of np.array array: initial parameters.
        """
        Parameterized.__init__(self)
        self.table = DictData(array)
    
    
