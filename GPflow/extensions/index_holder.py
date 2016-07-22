# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
from ..data_holders import DataHolder, DictData
from ..param import Parameterized

class ListData(DataHolder):
    """
    Object for treating multiple DataHolders (shapes can be different).
    
    [Typical usage]
    
    >>> list_of_arrays = [np.ones((2,1)), np.random.randn(3,1)]
    
    >>> list_data = ListData(list_of_arrays)
    
    >>> list_data[i] ---> returns a DataHolder containing list_of_arrays[i]
    

    To set data, either

    >>> list_data.set_data([np.ones((2,1)), np.random.randn(3,1)])

    >>> list_data[i].set_data(np.ones((2,1)))
    
    are possible.
    
    """
    def __init__(self, list_of_arrays, on_shape_change='raise', on_length_change='raise'):
        """
        list_of_arrays: list of numpy arrays of data
        """
        DataHolder.__init__(self)
        self.on_shape_change = on_shape_change
        self._dict_data = [DictData(ary, on_shape_change=self.on_shape_change)\
                                    for ary in list_of_arrays]
        
        assert on_length_change in ['raise', 'recompile']
        self.on_length_change= on_length_change
        

    def set_data(self, list_of_arrays):
        """
        modification of DictData.set_data to accept the list_of_arrays
        """
        if len(list_of_arrays) == len(self._dict_data):
            for i in range(len(self._dict_data)):
                self._dict_data[i].set_data(list_of_arrays[i])
        else:
            # if length changed
            if self.on_length_change == 'raise':
                raise ValueError("The length of this data-set must not change. \
                                  (perhaps make the model again from scratch?)")
            elif self.on_length_change == 'recompile':
                self._dict_data = [DictData(ary, on_shape_change=self.on_shape_change)\
                                    for ary in list_of_arrays]
                self.highest_parent._needs_recompile = True
                if hasattr(self.highest_parent, '_kill_autoflow'):
                    self.highest_parent._kill_autoflow()
            else:
                raise ValueError('invalid option')  # pragma: no-cover            

    # support indexing
    def __getitem__(self,index):
        return self._dict_data[index]
        
    def __len__(self):
        return len(self._dict_data)

    def __setstate__(self, d):
        # TODO
        pass
        """
        DataHolder.__setstate__(self, d)
        tf_array = tf.placeholder(dtype=self._array.dtype,
                                  shape=[None]*self._array.ndim,
                                  name=self.name)
        self._tf_array = tf_array
        """
        
    def get_feed_dict(self):
        feed_dict = {}        
        for d in self._dict_data:
            feed_dict.update(d.get_feed_dict())
        return feed_dict
        
    @property
    def _tf_array(self):
        """
        This property is prepared for Parameterized.__getattribute__ method.
        It will return a list of _tf_arrays.
        """
        return [d._tf_array for d in self._dict_data]
    
    def __iter__(self):
        # Overload __iter__ method
        return iter(self._dict_data)
        
        
class IndexHolder(Parameterized):
    """
    An object that stores index (n-length vector with int32 dtype).
    
    The permutation (self._permutation) and 
    inverse-permutation (self._inv_perm) are also stored, 
    
    e.g. 
    In case index = [1, 0, 2, 1, 0, 0]
    then self._permutation and self._inv_perm becomes
        self._permutation --> [[1, 4, 5], [0, 3], [2]]
        self._inv_perm    --> [3, 0, 5, 4, 1, 2]
    
    This object supports the following functionalties.
        
    + split(X)
        gives a list of tensor that is the split result of X according to 
        self.index
        
    + restore(list_of_X)
        gives a tensor X from a list_of_X in the original order.

    """
    
    def __init__(self, index, on_shape_change='pass', num_index=None):
        """
        - index : 1d np.array. The element is cast to int32. The index must be
                zero based.
        
        - on_shape_change : one of ('raise', 'pass', 'recompile'), which is the
                            functionality of DataHolder
        
        - num_index : defaults is np.max(index)+1. If index is a minibatch and 
                whole index has additional elements, it should be specified.
                    
        """
        Parameterized.__init__(self)
        self.index = DictData(np.array(index).astype(np.int32), on_shape_change=on_shape_change)
        
        if num_index is None:
            num_index = np.max(index)+1
        
        # define self._inv_perm
        self._inv_perm = DictData(np.ones((1), dtype=np.int32), on_shape_change='pass')
    
        # Define _permutation with dummy array
        self._permutation = ListData([np.ones((1), dtype=np.int32)] * num_index, \
                                        on_shape_change=on_shape_change,\
                                        on_length_change='recompile')
        # calculate self._permutation and self.inv_perm
        self.set_index(index)
        

    def set_index(self, index, num_index=None):
        """
        Method to set a new index.
        
        At the index update, self._backward_index should be also updated.
        
        # works with no _tf_mode
        """
        
        self.index.set_data(np.array(index).astype(np.int32))

        # construct self._permutation
        if num_index is None:
            num_index = len(self._permutation)
        perm = [None]*num_index
        for i in range(num_index):
            perm[i] = np.array(\
                [j for j in range(len(self.index.value)) if self.index.value[j] == i],\
                dtype=np.int32)

        self._permutation.set_data(perm)
        
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
            