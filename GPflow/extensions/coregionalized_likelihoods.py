# -*- coding: utf-8 -*-

from .. import likelihoods

class Likelihood(likelihoods.Likelihood):
    def __init__(self, list_likelihoods):
        """
        Construct likelihoods for coregionalized model.
        
        list_of_likelihoods: list of likelihood.
        """
        
        # TODO what about if number of labels changed?
        self.list_of_likelihoods = list_likelihoods
        
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
        val = []
        for (fmu, fvar, lik) in zip(Xnew.split(Fmu), Xnew.split(Fvar), \
                                                    self.list_of_likelihoods):
            val.append(lik.predict_mean_and_var(fmu, fvar))
        return Xnew.restore(val)
        
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
        