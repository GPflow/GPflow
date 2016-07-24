# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 18:53:16 2016

@author: keisukefujii
"""
import numpy as np
from .model import Coregionalized_GPModel
from ..model import GPModel
from ..param import Param
from .. import transforms
from .. import vgp
from .labeled_data import LabeledData
from .mean_functions import MeanFunction as coregionalized_mean_function
from ..mean_functions import Zero

class VGP(Coregionalized_GPModel, vgp.VGP):
    """
    Coregionalized VGP.
    
    This method inheritates from Coregionalized_GPModel and vgp.VGP.
    
    Coregionalized_GPModel provides some methods relating to the AutoFlow
    wrapping.
    """
    def __init__(self, X, Y, label, kern, likelihood, 
                     mean_function=None, num_labels=None, num_latent=None):
        """
        X is a data matrix, size N x D
        Y is a data matrix, size N x R
        label is a label for the data, size N
        kern should be one of coregionalized_kernels.
        """
        X = LabeledData((X, label), on_shape_change='pass', num_labels=num_labels)
        Y = LabeledData((Y, label), on_shape_change='pass', num_labels=num_labels)
        
        # If mean_function is None, Zero likelihoods are assumed.
        if mean_function is None:
            mean_function = coregionalized_mean_function.MeanFunction(\
                                                     [Zero()]*X.num_labels)
        
        # initialize GPModel rather than gpr
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function)

        self.num_data = X.shape[0]
        self.num_latent = num_latent or Y.shape[1]
        self.q_alpha = Param(np.zeros((self.num_data, self.num_latent)))
        self.q_lambda = Param(np.ones((self.num_data, self.num_latent)),
                              transforms.positive)
        
