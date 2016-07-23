# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ..data_holders import DictData, DataHolderList
from ..param import Parameterized

        
class LabelHolder(Parameterized):
    """
    An object that stores label.
    self.label: is 1-dimensional array with label.shape[0] = data.shape[0]
    (n-length vector with int32 dtype)
    
    For the convenience, permutation (self._permutation) and 
    inverse-permutation (self._inv_perm) are also stored, 
    
    e.g. 
    In case label = [1, 0, 2, 1, 0, 0]
    then self._permutation and self._inv_perm becomes
        self._permutation --> [[1, 4, 5], [0, 3], [2]]
        self._inv_perm    --> [3, 0, 5, 4, 1, 2]
    
    This object supports the following functionalties.
        
    + split(X)
        gives a list of tensor that is the split result of X according to 
        self.label.
        
    + restore(list_of_X)
        gives a tensor X from a list_of_X in the original order.

    """
    
    def __init__(self, label, on_shape_change='raise', num_labels=None):
        """
        - array: 2d np.array for data.
        
        - label : 1d np.array. The element is cast to int32. The label must be
                zero based.
        
        - on_shape_change : one of ('raise', 'pass', 'recompile'), which is the
                            functionality of DataDict
        
        - num_labels : defaults is np.max(label)+1. If label is a minibatch and 
                whole labels have additional elements, it should be specified.
                    
        """
        Parameterized.__init__(self)
        self.label = DictData(np.array(label).astype(np.int32), on_shape_change=on_shape_change)
        
        if num_labels is None:
            num_labels = np.max(label)+1
        
        # define self._inv_perm
        self._inv_perm = DictData(np.ones((1), dtype=np.int32), on_shape_change='pass')
    
        # Define _permutation with dummy array
        self._permutation = DataHolderList()
        for i in range(num_labels):
            self._permutation.append(DictData(np.ones((1), dtype=np.int32), on_shape_change='pass'))

        # calculate self._permutation and self.inv_perm
        self.set_label(label)
        

    def set_label(self, label, num_labels=None):
        """
        Method to set a new data and label.
        
        At the label update, self._permutation and _inv_perm should be also updated.
        
        # works with no _tf_mode
        """
        self.label.set_data(np.array(label).astype(np.int32))

        # construct self._permutation
        if num_labels is None:
            num_labels = len(self._permutation)
        perm = [None]*num_labels
        for i in range(num_labels):
            perm[i] = np.array(\
                [j for j in range(len(self.label.value)) if self.label.value[j] == i],\
                dtype=np.int32)

        self._permutation.set_data(perm)
        
        # construct self._inv_perm
        perm_concat = np.concatenate([p.value for p in self._permutation])
        inv_perm = np.argsort(perm_concat)
        self._inv_perm.set_data(np.array(inv_perm, dtype=np.int32))

    @property
    def num_labels(self):
        return len(self._permutation)
                    
    def split(self, X):
        """
        split into a tensor X according to self.label.
        
        X is DataHolder, where X.value is a 2-D np.array
          or tf.PlaceHolder of 2-D tensor.
        """
        rslt = []
        if not self._tf_mode:
            # split X.value
            for perm in self._permutation:
                rslt.append(X.value[perm.value, :])
        else:
            for perm in self._permutation:
                # access to _tf_array directly because 
                # Parameterized.__getattribute does not work for list of 
                # dataholders
                rslt.append(tf.gather(X, perm))     
        return rslt
    
    def restore(self, list_of_X):
        """
        restore a tensor X from a list_of_X in the original order.
        """
        if not self._tf_mode:
            if isinstance(list_of_X[0], np.ndarray):
                 return np.concatenate(list_of_X)[self._inv_perm.value]
            else:
                return np.concatenate([x.value for x in list_of_X])[self._inv_perm]
        else:
            return tf.gather(tf.concat(0,list_of_X), self._inv_perm)

    def setup_diag(self, list_of_X):
        """
        
        """

class LabeledData(LabelHolder):
    """
    Object that stores data and (only one kind of) label.
    """
    def __init__(self, array, label, on_shape_change='raise', num_labels=None):
        LabelHolder.__init__(self, label, on_shape_change, num_labels)
        # the length of array and label should be the same
        assert(len(array) == len(label))
        self.data = DictData(array, on_shape_change=on_shape_change)
    
    def set_data(self, array, label, num_labels=None):
        assert(len(array) == len(label))
        LabelHolder.set_label(self, label, num_labels)
        self.data.set_data(array)
