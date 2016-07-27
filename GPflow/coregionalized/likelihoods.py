# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 2016

@author: keisukefujii
"""

from .. import likelihoods
from ..param import ParamList

class Likelihood(likelihoods.Likelihood):
    def __init__(self, list_of_likelihoods):
        """
        Construct likelihoods for coregionalized model.
        
        list_of_likelihoods: list of likelihood.
        """
        
        # TODO what about if number of labels changed?
        self.list_of_likelihoods = ParamList(list_of_likelihoods)
        
    
    def logp(self, F, Y):
        """
        F : tf.variable
        Y : Labeled data
        """
        val = []
        for (f, y, lik) in zip(Y.split(F), Y.split(Y.data), self.list_of_likelihoods):
            val.append(lik.logp(f,y))
        return Y.restore(val)

    def conditional_mean(self, F):
        # TODO LabelData should be passed in some way...
        raise NotImplementedError
        
    def conditional_variance(self, F):
        # TODO LabelData should be passed in some way...
        raise NotImplementedError
        
    def predict_mean_and_var(self, Fmu, Fvar, Xnew):
        # Xnew is added to the original methods
        mu = []
        var = []
        for (fmu, fvar, lik) in zip(Xnew.split(Fmu), Xnew.split(Fvar), \
                                                    self.list_of_likelihoods):
            mu_tmp, var_tmp = lik.predict_mean_and_var(fmu, fvar)
            mu.append(mu_tmp)
            var.append(var_tmp)
        return Xnew.restore(mu), Xnew.restore(var)
        
    def predict_density(self, Fmu, Fvar, Y):
        val = []
        for (fmu, fvar, y, lik) in zip(Y.split(Fmu), Y.split(Fvar), \
                                Y.split(Y.data), self.list_of_likelihoods):
            val.append(lik.predict_density(fmu, fvar, y))
        return Y.restore(val)
        
    def variational_expectations(self, Fmu, Fvar, Y):
        val = []
        for (fmu, fvar, y, lik) in zip(Y.split(Fmu), Y.split(Fvar), \
                                Y.split(Y.data), self.list_of_likelihoods):
            val.append(lik.variational_expectations(fmu, fvar, y))
        return Y.restore(val)

    def __iter__(self):
        return iter(self.list_of_likelihoods)

  