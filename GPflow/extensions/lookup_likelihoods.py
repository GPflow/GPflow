# -*- coding: utf-8 -*-

from GPflow import densities
import tensorflow as tf
import numpy as np
from GPflow import likelihoods
from GPflow.param import Parameterized, Param
from GPflow import transforms
from lookup_param import LookupParam, LookupDataHolder, IndexPicker
hermgauss = np.polynomial.hermite.hermgauss

class Likelihood(likelihoods.Likelihood):
    """
    Coregionalized version of likelihood.
    """