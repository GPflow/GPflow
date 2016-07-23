# -*- coding: utf-8 -*-

from ..kernels import Kern


class LookupKern(Kern):
    """
    The basic kernel class that enables lookup from data or parameters.
    
    The arugument for self.K(X) and self.K_diag(X) should be LabelData.
    
    The lookup kernel should be implemented in inheriting class.
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
            return self._K_train(X.label)
        else:
            return self._K_pred(X.label, X2.label)
        
    def Kdiag(self, X):
        return self._K_diag(X.label)


    def _K_train(self, label):
        """
        Kernel values for the training.
        It corresponds Kern.K(X).
        """
        raise NotImplementedError

    def _K_pred(self, label, label2):
        """
        Kernel values for the prediction.
        It corresponds Kern.K(X, X2).
        """
        raise NotImplementedError
        
    def _K_diag(self, label):
        """
        Diagonal part of _K_train.
        """
        raise NotImplementedError


    