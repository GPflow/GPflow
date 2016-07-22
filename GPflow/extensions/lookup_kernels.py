# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from GPflow.kernels import Kern
from GPflow.param import Param, Parameterized, AutoFlow
from GPflow import transforms
from .lookup_params import LookupParam, IndexPicker

class LookupKern(Kern):
    """
    The basic kernel class that enables lookup from data and parameters.
    """
    def __init__(self, input_dim, index_dim= -1):
        """
        index_dim is an integer.
        
        The argument tensor X has to have index_dim, which is used for the 
        lookup from table, i.e. table are looked up by
        table[x[:, index_dim]]
        
        The generic '_to_index' function is implemented to make a 1-d tensor
        self._to_index(x) 
        which is like x[:, index_dim]
        """
        Kern.__init__(self, input_dim=1)
        self.picker = IndexPicker(input_dim=input_dim, index_dim=index_dim)
        
    def _to_index(self, X):
        return self.picker.to_index(X)
        

class LinearCoregionalizedKernel(LookupKern):
    """
    Kernel object that realizes the coregionalized model.
    
    This kernel has two parameters, w and kappa.
    
    Typical usage for coregionalized data pair (x1, y1) and (x2, y2)
    
    >>> X = DataHolder([[x_, 0] for x_ in x0] + [[x_, 1] for x1 in x1])
    
    >>> Y = DataHolder([[y_, 0] for y_ in y0] + [[y_, 1] for y1 in y1])

    RBF kernel for the first dimension of X
    >>> Kr = RBF(input_dim=2, active_dims=0)

    Lookupkernel for the second dimension of X
    >>> Kb = LinearCoregionalizedKernel(index_dim = 1, number_of_tasks=2, rank = 1)
    
    The total Kronecker's multiple of Kr and Kb is
    >>> K = Kb*Kr
    """
    def __init__(self, index_dim, number_of_tasks, rank, w=None, kappa=None):
        LookupKern.__init__(self, input_dim = index_dim+1, index_dim=index_dim)
    
        if w is not None:
            assert(w.shape[0] == number_of_tasks)
            assert(w.shape[1] == rank)
            self.w = LookupParam(w)
        else:
            self.w = LookupParam(np.ones((number_of_tasks, rank))*0.5)
        
        if kappa is not None:
            assert(kappa.shape[0] == number_of_tasks)
            self.kappa = LookupParam(kappa, transform=transforms.positive)
        else:
            self.kappa = LookupParam(np.ones(number_of_tasks))
            
    def K(self, X, X2=None):
        X = self._to_index(X)
        if X2 is None:
            return tf.matmul(self.w(X), self.w(X), transpose_b=True) \
                    + self.kappa.diag(X,X)
        else:
            X2 = self._to_index(X2)
            return tf.matmul(self.w(X), self.w(X2), transpose_b=True) \
                    + self.kappa.diag(X, X2)
    
    def Kdiag(self, X):
        return tf.diag(tf.diag_part(self.K(X, X)))
        
    