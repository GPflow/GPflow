# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 2016

@author: keisukefujii
"""
import numpy as np
import tensorflow as tf
from ..data_holders import DataHolder, DictData, DataHolderList
from ..param import Parameterized

class LabelHolder(DataHolder):
    """
    An object that stores label.
    self._label: is 1-dimensional array with label.shape[0] = data.shape[0]
    (n-length vector with int32 dtype)
    
    For the convenience, permutation (self._permutation, DataHolderList) and 
    inverse-permutation (self._inv_perm, DictData) are also stored.
    
    
    e.g. 
    Consider label = [1, 0, 2, 1, 0, 0]
    then self._permutation and self._inv_perm becomes
        self._permutation --> [[1, 4, 5], [0, 3], [2]]
        self._inv_perm    --> [3, 0, 5, 4, 1, 2]
    
    This object supports the following functionalties.
        
    + split(X)
        gives a list of tensor that is the split result of X according to 
        self._label.
        
    + restore(list_of_X)
        gives a tensor X from a list_of_X into the original order.

    """
    
    def __init__(self, label, on_shape_change='raise', num_labels=None):
        """
        - label : 1d np.array. The element is cast to int32. The label must be
                zero based.
        
        - on_shape_change : one of ('raise', 'pass', 'recompile'), which is the
                            functionality of DataDict
        
        - num_labels : defaults is np.max(label)+1. If label is a minibatch and 
                whole labels have additional elements, it should be specified.
                    
        For the convenience, 
        self.label is provided to access self._label.tf_array,
        
        """
        DataHolder.__init__(self)
        if num_labels is None:
            num_labels = np.max(label)+1
        
        # set the label
        self._label = DictData(np.array(label, dtype=np.int32), \
                                                on_shape_change=on_shape_change)
        
        # define self._inv_perm
        self._inv_perm = DictData(np.ones((1), dtype=np.int32), \
                                                        on_shape_change='pass')
    
        # Define _permutation with dummy array
        self._permutation = DataHolderList()
        for i in range(num_labels):
            self._permutation.append(DictData(np.ones((1), dtype=np.int32), \
                                                        on_shape_change='pass'))

        # calculate self._permutation and self.inv_perm
        self.set_label(label)
        
    def set_label(self, label):
        """
        Method to set a new label.
        
        At the label update, self._permutation and _inv_perm should be also updated.
        
        # works with no _tf_mode
        """
        # with cast to np.int32
        self._label.set_data(np.array(label).astype(np.int32))

        # check if (0 < label < num_label).all()
        if np.min(label) < 0:
            raise IndexError('Invalid label was provided. np.min(label)=' + \
                                                str(np.min(label))+' < 0')
        if self.num_labels <= np.max(label):            
            raise IndexError('Invalid label was provided. np.max(label)=' + \
                str(np.min(label))+' >= num_labels('+str(self.num_labels)+')')
            
        # construct self._permutation
        perm = [None]*self.num_labels
        for i in range(self.num_labels):
            perm[i] = np.array(\
                [j for j in range(len(self._label.value)) if self._label.value[j] == i],\
                dtype=np.int32)

        self._permutation.set_data(perm)
        
        # construct self._inv_perm
        perm_concat = np.concatenate([p.value for p in self._permutation])
        inv_perm = np.argsort(perm_concat)
        self._inv_perm.set_data(np.array(inv_perm, dtype=np.int32))

    def set_data(self, label):
        self.set_label(label)

    @property
    def num_labels(self):
        return len(self._permutation)
                    
    def split(self, X):
        """
        split into a tensor X according to self._label.
        """
        rslt = []
        for perm in self._permutation:
            # access to _tf_array directly because 
            # Parameterized.__getattribute does not work for list of 
            # dataholders
            rslt.append(tf.gather(X, perm._tf_array))
        return rslt
    
    def restore(self, list_of_X):
        """
        restore a tensor X from a list_of_X in the original order.
        """
        return tf.gather(tf.concat(0,list_of_X), self._inv_perm._tf_array)
        
    @property
    def _tf_array(self):
        """
        Behave as a DataHolder, but returns this object.
        """        
        return self
        
    def get_feed_dict(self):
        # DataHolder members are label, _permutation, _inv_perm
        feed_dict = self._label.get_feed_dict()
        feed_dict.update(self._permutation.get_feed_dict())
        feed_dict.update(self._inv_perm.get_feed_dict())
        return feed_dict
        
    @property
    def label(self):
        """
        Shortcut property for the convenience.
        """
        return self._label._tf_array


class LabeledData(LabelHolder):
    """
    Object that stores data and (only one kind) label.
    """
    def __init__(self, X_label_tuple, on_shape_change='raise', num_labels=None):
        """
        X_label_tuple is a 2-element tuple,
        X_label_tuple[0] : data as 2d-np.array 
        X_label_tuple[1] : label as 1d-np.array
        """
        # the length of array and label should be the same
        if isinstance(X_label_tuple[1], np.ndarray):
            assert(X_label_tuple[0].shape[0] == X_label_tuple[1].shape[0])
        elif isinstance(X_label_tuple[1], list):
            assert(X_label_tuple[0].shape[0] == len(X_label_tuple[1]))
        else:
            raise TypeError('X_label_tuple[1] should be a 1-d np.array or list')

        self._data = DictData(X_label_tuple[0], on_shape_change=on_shape_change)
        LabelHolder.__init__(self, X_label_tuple[1], on_shape_change, num_labels)
    
    
    def set_data(self, X_label_tuple):
        # the length of array and label should be the same
        if isinstance(X_label_tuple[1], np.ndarray):
            assert(X_label_tuple[0].shape[0] == X_label_tuple[1].shape[0])
        elif isinstance(X_label_tuple[1], list):
            assert(X_label_tuple[0].shape[0] == len(X_label_tuple[1]))
        else:
            raise TypeError('X_label_tuple[1] should be a 1-d np.array or list')
        self._data.set_data(X_label_tuple[0])
        LabelHolder.set_data(self, X_label_tuple[1])

    def get_feed_dict(self):
        # additional DataHolder member of this class is self.data
        feed_dict = LabelHolder.get_feed_dict(self)
        feed_dict.update(self._data.get_feed_dict())
        return feed_dict

    @property
    def data(self):
        """
        Shortcut property for the convenience.
        """
        return self._data._tf_array
    
    @property
    def shape(self):
        """
        shape property returns the shape of the data.
        """
        return self._data.shape
        
    #---- oberriding the operators ----
    def __sub__(self, obj):
        # assuming obj is always tf.variable
        return self.data - obj
        
    
