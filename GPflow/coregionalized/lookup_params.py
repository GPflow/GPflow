# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 2016

@author: keisukefujii
"""
import tensorflow as tf
from ..param import Param, Parameterized
from ..data_holders import DictData
from .. import transforms, tf_hacks
            

class LookupParam(Parameterized):
    """
    Parameter object that accepts index.
    
    To instanciate the object, an initial parameter np.arrays should be 
    provided.
    
    >>> lookup_param = LookupParam(init_array)
    
    This object stores a GPflow.Param (lookup_param._table) with 
    first (several) dimension(s) for the indexed access.
    
    To access the GPflow.param, following methods are provided,
    
    with no tf_mode,
    >>> lookup_param(i)
    returns lookup_param._table.value[i]
    
    >>> lookup_param(i,j)
    returns lookup_param._table.value[i,j]
    
    with tf_mode, 
    1-dimensional-like integer tensor x (usually, LabeledData.label) should be 
    passed,

    >>> lookup_param(x)
    return is like [lookup_param._table[x_i] for i in range(len(x))]    

    >>> lookup_param(x,x)
    return is like [[lookup_param._table[x_i, x_j] for j in range(len(x))] for k in range(len(x))]
    
    
    If lookup_para._table is 1-d, an additional method is available,
    
    >>> lookup_param.diag(i, j)
    returns if diag(lookup_param._table)[i,j]
    
    
    For the fix,
    >>> lookup_param.fixed = True 
    >>> lookup_param.fixed = False
    can be used as usual Parameterized object.
    
    To get the whole part of _table
    >>> current_table = lookup_param.table.value

    To replace _table,
    >>> lookup_param.table = np.array
    
    """
    
    def __init__(self, array, transform=transforms.Identity()):
        """
        :param np.array or list of np.array array: initial parameters.
        """
        Parameterized.__init__(self)
        self.table = Param(array, transform)
        self.shape = self.table.value.shape
    

    def __setattr__(self, key, value):
        """
        When new value is updated, self.shape should be also updated.
        """
        Parameterized.__setattr__(self, key, value)
        if key == 'table':
            self.shape = self.table.value.shape

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
            
            return tf.transpose(tf.gather(\
                   tf.transpose(tf.gather(tf_hacks.diag_1stdim(self.table) , i))\
                                                                           , j))
        else:
            if i==j:
                return self.__call__(i)
            else:
                return 0.0

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
    
    
