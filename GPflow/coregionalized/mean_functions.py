# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 2016

@author: keisukefujii
"""

from .. import mean_functions
from ..param import ParamList

class MeanFunction(mean_functions.MeanFunction):
    def __init__(self, list_of_mean_functions):
        """
        Construct likelihoods for coregionalized model.
        
        list_of_mean_functions: list of likelihood used for each task.
        """
        # TODO what about if number of labelsk changed?
        self.list_of_mean_functions = ParamList(list_of_mean_functions)

    def __call__(self, X):
        val = []
        for (x, mean) in zip(X.split(X.data), self.list_of_mean_functions._list):
            val.append(mean(x))
        return X.restore(val)

# TODO added the str() method.