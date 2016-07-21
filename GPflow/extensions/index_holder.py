# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ..param import DataHolder, Param, Parameterized
from .. import transforms

# TODO inherite DataHolder class

class IndexHolder(Parameterized):
    """
    An object that stores index (n-length vector with int32 dtype).
    
    The backward index (see restore method below) is also stored, 
    self._permutation, as a list of DataHolders.
    
    e.g. 
    In case index = [1, 0, 2, 1, 0, 0]
    then self._permutation and self._inv_perm becomes
        self._permutation = [[1, 4, 5], [0, 3], [2]]
        self._inv_perm = [3, 0, 5, 4, 1, 2]
    
    This object supports the following functionalties.
        
    + split(X)
        gives a list of tensor that is the split result of X according to 
        self.index
        
            
        
    + restore(list_of_X)
        gives a tensor X from a list_of_X in the original order.
        
    To enable compilation of list of DataHolders, some Parameterized 
    methods are overwritten.
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
        self.index = DataHolder(np.array(index).astype(np.int32), on_shape_change=on_shape_change)
        
        if num_index is None:
            num_index = np.max(index)+1
        self.num_index=num_index
        
        # define self._permutation
        self._permutation = []
        for i in range(num_index):
            self._permutation.append(\
                DataHolder(np.ones((1), dtype=np.int32), on_shape_change='pass'))
        # define self._inv_perm
        self._inv_perm = DataHolder(np.ones((1), dtype=np.int32), on_shape_change='pass')
    
        # calculate self._permutation and self.inv_perm
        self.set_index(index)
        

    def set_index(self, index, num_index=None):
        """
        Method to set a new index.
        
        At the index update, self._backward_index should be also updated.
        
        # works with no _tf_mode
        """
        if num_index is not None and num_index != self.num_index:
            # raise the recompilation flag.
            if hasattr(self.highest_parent, '_needs_recompile'):
                self.highest_parent._needs_recompile = True
            self.num_index = num_index
            # define self._backward_index
            self._permutation = []
            for i in range(self.num_index):
                self._permutation.append(\
                    DataHolder(np.ones((1), dtype=np.int32), on_shape_change='pass'))
        
        self.index.set_data(np.array(index).astype(np.int32))
        # construct self._permutation
        for i in range(self.num_index):
            perm = [j for j in range(len(self.index.value)) if self.index.value[j] == i]
            self._permutation[i].set_data(np.array(perm, dtype=np.int32))
        
        # construct self._inv_perm
        perm_concat = np.concatenate([p.value for p in self._permutation])
        inv_perm = np.argsort(perm_concat)
        self._inv_perm.set_data(np.array(inv_perm, dtype=np.int32))
                    
    def split(self, X):
        """
        split into a tensor X according to self.index.
        
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
                rslt.append(tf.gather(X, perm._tf_array))     
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
        
    @property
    def data_holders(self):
        """
        Overwrite Parameterized method to enable compilation of list of DataHolders.
        """
        return super(IndexHolder, self).data_holders + self._permutation


'''
    def _begin_tf_mode(self):
        """
        Overwrite to enable tf_mode also for self._permutation
        """
        Parameterized._begin_tf_mode(self)
        for perm in self._permutation:
            perm._begin_tf_mode()

    def _end_tf_mode(self):
        Parameterized._end_tf_mode(self)
        for perm in self._permutation:
            perm._end_tf_mode()
'''            
        
        
'''        
 class Data_list(Parameterized):
    """
    Object that stores a list of DataHolders.
    
    It overloads some Parameterized's method so that the list is apparent in 
    tf_mode.

    >> dlist = Data_list(list_or_arrays)
    
    To access the each element, (without tf_mode)

    >>  for d in dlist:
    >>      d.value
    
    or

    >>  for i in range(len(dlist)):
    >>      dlist[i].value
    
    """
    def __init__(self, list_of_arrays, on_shape_change='raise'):
        # Initialize for parent class. 
        super(Data_list, self).__init__()
        
        self._list = []
        for ary in list_of_arrays:
            self._list.append(DataHolder(ary , on_shape_change))
        
        self._i = 0


    @property
    def data_holders(self):
        """
        Overwrite data_holders method.
        The only dataholders in this object are self._list
        """
        return self._list
    
    
    def __getattribute__(self, key):
        """
        If key is "_list" and in tf_mode, the returns the list of _tf_array.
        Otherwise, returns the super.__getattribute__
        """
        if key == "_list" and object.__getattribute__(self, '_tf_mode'):
            return [l._tf_array for l in object.__getattribute__(self, key)]
        return super(Data_list, self).__getattribute__(key)
        
    def __iter__(self):
        # initialize index
        self._i = 0
        return self

    def __next__(self):
        if self._i == len(self._list):
            raise StopIteration()
        self._i += 1
        return self._list[self._i - 1]

    def next(self):
        # for python 2 support
        return self.__next__()
    
    def __getitem__(self, index):
        return self._list[index]    
        
    def __len__(self):
        return len(self._list)
    
    def concat(self):
        """
        Returns a stacked array.
        Without _tf_mode, returns stacked np.array
        With _tf_mode, returns the packed tensor
        """
        if not self._tf_mode:
            return np.vstack([l.value for l in self._list])
        else:
            return tf.concat(0, self._list)
'''                   