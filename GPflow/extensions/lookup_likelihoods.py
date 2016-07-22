# -*- coding: utf-8 -*-

from GPflow import densities
import tensorflow as tf
import numpy as np
from .. import likelihoods
from ..likelihoods import hermgauss
from ..param import Parameterized, Param
from .. import transforms
from index_holder import IndexHolder

class Likelihood(likelihoods.Likelihood):
    """
    Coregionalized version of likelihood.
    
    self.set_index(index) should be called before executing any tensorflow function.
    """
    def __init__(self):
        likelihoods.Likelihood(self)
        
    def set_index(self, index, index2=None):
        """
        index: IndexHolder with appropriate order and dimension.
        """
        self.I= index

class gaussian(Likelihood):
    def __init__(self, number_of_tasks):
        