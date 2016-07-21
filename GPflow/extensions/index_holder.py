# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from .param import DataHolder, Param, Parameterized
from . import transforms

class IndexHolder(Parameterized):
    """
    An object that stores index (n-length vector with int32 dtype).
    
    The index is stored as DataHolder (self.index).
    
    This object supports the following functionalties.
        
    + split(X, index_dim)
        gives a list of tensor that is the split result of X according to 
        self._forward_index[:, index_dim]
        
    + restore(list_of_X, index_dim)
        gives a tensor X from a list_of_X in the original order.
        
    """
    
    def __init__(self, index, on_shape_change='raise', num_index=None):
        """
        - index : 1d np.array. The element is cast to int32. The index must be
                zero based.
        
        - on_shape_change : one of ('raise', 'pass', 'recompile'), which is the
                            functionality of DataHolder
        
        - num_index : defaults is np.max(index)+1. If index is a minibatch and 
                whole index has additional elements, it should be specified.
                    
        """
        Parameterized.__init__(self)
        self.index = DataHolder(index.as_type(np.int32), on_shape_change=on_shape_change)
        self.num_index=num_index
        

    def set_index(self, index, num_index=None):
        """
        Method to set a new index.
        
        At the index update, self._backward_index should be also updated.
        """
        if num_index is not None and num_index != self.num_index:
            # raise the recompilation flag.
            if hasattr(self.highest_parent, '_needs_recompile'):
                self.highest_parent._needs_recompile = True
            
            self.num_index = num_index
            
        
        