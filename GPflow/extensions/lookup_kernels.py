# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
from ..param import AutoFlow
from ..kernels import Kern, Constant
from .. import transforms
from .lookup_params import LookupParam


class LookupKern(Kern):
    """
    The basic kernel class that enables lookup from data or parameters.
    
    self.set_index(index, index2) should be called before calling self.K and
    self.Kdiag.
    
    The arguments X and X2 in K and Kdiag are just a dummy.
    """
    def __init__(self):
        # input dim for kernel is dummy.
        Kern.__init__(self, input_dim = 1)

 
    def set_index(self, index, index2=None):
        """
        index, index2: IndexHolder with appropriate order and dimension.
        
        If index2 is used if X2 is not None in self.K
        """
        self._I= index
        self._I2 = index2


    def K(self, X, X2=None):
        if X2 is None:
            return self._K_train(self._I)
        else:
            return self._K_pred(self._I, self._I2)
        
    def Kdiag(self, X):
        return self._K_diag(self._I)


    def _K_train(self, I):
        """
        Kernel values for the training.
        It corresponds Kern.K(X).
        """
        raise NotImplementedError

    def _K_pred(self, I):
        """
        Kernel values for the prediction.
        It corresponds Kern.K(X, X2).
        """
        raise NotImplementedError
        
    def _K_diag(self, I):
        """
        Diagonal part of _K_train.
        """
        raise NotImplementedError


class CoregionalizedKern(LookupKern):
    """
    """
    def __init__(self):
        LookupKern.__init__(self)

    def construct_Kronecker(self, base_kern=None):
        """
        Genekrate Kronecker's multiple.
        base_kern is Kern object.
        """
        if base_kern is None:
            base_kern = Constant(input_dim=1)
            base_kern.variance.fixed = True
        self.base_kern = base_kern 
        
    def K(self, X, X2=None):
        if X2 is None:
            return self._K_train(self._I) * self.base_kern.K(X)
        else:
            return self._K_pred(self._I, self._I2) * self.base_kern.K(X, X2)
        
    def Kdiag(self, X):
        return self._K_diag(self._I) * self.base_kern.K(X)
        

    # TODO for addition and multiplication  
    
    # TODO autoflow
    """
    @AutoFlow((tf.float64, [None, None]), (tf.float64, [None, None]))
    def compute_K(self, X, Z):
        return self.K(X, Z)

    @AutoFlow((tf.float64, [None, None]))
    def compute_K_symm(self, X):
        return self.K(X)
    """

class LinearCoregionalizedKernel(CoregionalizedKern):
    """
    Kernel object that realizes the coregionalized model.

    This kernel has two parameters, w and kappa.
    The value of kernel for each task is
    [w * w^t](i,j) + kappa(i)delta(i)
    
    Since this kernel has lookup feature, IndexHolder has to be provided.
    
    A typical usage for coregionalized data pair (x1, y1) and (x2, y2)
    
    1. Prepare concatenated data X and Y
    
    >>> X = DictData(np.concatenate([x1, x2]))

    >>> Y = DictData(np.concatenate([y1, y2]))

    2. Prepare task index as IndexHolder with length len(x1)+len(x2), zero base
    and continuous incrementation.
    
    >>> I = IndexHolder(np.concate([np.zeros(len(x1)), np.ones(len(x2))]))
    
    3. Set the IndexHolder into this kernel.
    
    >>> Kcor = LinearCoregionalizedKernel(Kern, num_tasks)
    
    >>> Kcor.set_index(I)
    
    4. The base kernel should be prepared, e.g. RBF kernel
    
    >>> Kr = RBF(input_dim=1)
    
    5. The coregionalized kernel can be constructed by Kronecker method
    
    >>> K = Kcor.Kronecker(Kr)
    
    resulting in the Kronecker's multiple of Kr and Kcor.
    
    """
    def __init__(self, number_of_tasks, rank, w=None, kappa=None):
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
        
        CoregionalizedKern.__init__(self)
        
    def _K_train(self, I):
        return tf.matmul(self.w(I.index), self.w(I.index), transpose_b=True) \
            + self.kappa.diag(I.index, I.index)

    def _K_pred(self, I, I2):
        return tf.matmul(self.w(I.index), self.w(I2.index), transpose_b=True) \
            + self.kappa.diag(I.index, I2.index)
        
    def _K_diag(self, I):
        return tf.diag_part(self._K_train(I))
            
    