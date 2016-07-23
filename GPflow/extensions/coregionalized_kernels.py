# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from ..param import AutoFlow
from .. import kernels
from .. import transforms
from .lookup_params import LookupParam
from .lookup_kernels import LookupKern

class Kern(LookupKern):
    """
    Base kernel object that enables to coregionalization.
    
    The arguments of self.K(X) and self.Kdiag(X) should be LabelData.    
    
    The Kronecker's multiplication is realized by using LabelData.data and 
    LabelData.label


    A typical usage for coregionalized data pair (x1, y1) and (x2, y2) are
    
    1. Prepare base kernel, e.g. RBF kernel

    >>> base_kern = RBF(input_dim=1)
    
    2. Construct coregionalized kernel (e.g. LinearCoregionalizedKernel)
    
    >>> K = LinearCoregionalizedKernel(base_kern, number_of_task=2, rank=1)

    3. Prepare concatenated data X and Y
    
    >>> X = np.concatenate([x1, x2])

    >>> Y = np.concatenate([y1, y2])

    4. Prepare task label as 1-d np.array with length len(x1)+len(x2), 
    zero base and continuous incrementation.
    
    >>> I = np.concate([np.zeros(len(x1)), np.ones(len(x2))])
    
    5. Prepare Labeled data for X
    
    >>> Xl = LabeledData(X, I, on_shape_change)

    6. Get the kernel values!
    
    >>> K.K(Xl)
            
    """
    def __init__(self, base_kern=None):
        """
        baser_kern: (normal) Kern object.
        Params in base_kern should be appropriately fixed.
        """
        LookupKern.__init__(self)

        if base_kern is None:
            base_kern = kernels.Constant(input_dim=1)
            base_kern.variance.fixed = True
        self.base_kern = base_kern 
        
    def K(self, X, X2=None):
        if X2 is None:
            return self._K_train(X.label) * self.base_kern.K(X.data)
        else:
            return self._K_pred(X.label, X2.label) * self.base_kern.K(X.data, X2.data)
        
    def Kdiag(self, X):
        return self._K_diag(X.label) * self.base_kern.K(X.data)
        

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

class Linear(Kern):
    """
    Kernel object that realizes the coregionalized model.

    This kernel has two parameters, w and kappa.
    The value of kernel for each task is
    [w * w^t](i,j) + kappa(i)delta(i)
    
    Since this kernel has lookup feature, IndexHolder has to be provided.
    """
    def __init__(self, base_kern, number_of_tasks, rank, w=None, kappa=None):
        if w is not None:
            assert(w.shape[0] == number_of_tasks)
            assert(w.shape[1] == rank)
            self.w = LookupParam(w)
        else:
            self.w = LookupParam(np.ones((number_of_tasks, rank))*0.5)
        
        if kappa is not None:
            assert(kappa.shape[0] == number_of_tasks)
            # make sure kappa is a 1d vector
            self.kappa = LookupParam(np.squeeze(kappa), transform=transforms.positive)
        else:
            self.kappa = LookupParam(np.ones(number_of_tasks), transform=transforms.positive)
        
        Kern.__init__(self, base_kern)
        
    def _K_train(self, label):
        return tf.matmul(self.w(label), self.w(label), transpose_b=True) \
            + self.kappa.diag(label, label)

    def _K_pred(self, label, label2):
        return tf.matmul(self.w(label), self.w(label2), transpose_b=True) \
            + self.kappa.diag(label, label2)
        
    def _K_diag(self, label):
        return tf.diag_part(self._K_train(label))
            
