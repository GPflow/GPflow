from typing import List

import tensorflow as tf

from .base import Likelihood


class MultiOutputLikelihood(Likelihood):
    def __init__(self, likelihoods: List[Likelihood]):
        assert len(likelihoods) == 2
        assert all(l.latent_dim == 2 for l in likelihoods)
        assert all(l.observation_dim == 1 for l in likelihoods)

        self.likelihoods = likelihoods
        super().__init__(latent_dim=sum(self.latent_dims), observation_dim=len(likelihoods))

    @property
    def latent_dims(self):
        return [l.latent_dim for l in self.likelihoods]

    def _log_prob(self, *args, **kw):
        pass

    def _split_fs(self, F):
        # TODO hardcoded for now
        F1 = F[..., :2]
        F2 = F[..., 2:]
        return [F1, F2]

    def _variational_expectations(self, Fmu, Fvar, Y):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        Y_splits = [Y[..., 0], Y[..., 1]]
        ve0 = self.likelihoods[0]._variational_expectations(
            Fmu_splits[0], Fvar_splits[0], Y_splits[0]
        )
        ve1 = self.likelihoods[1]._variational_expectations(
            Fmu_splits[1], Fvar_splits[1], Y_splits[1]
        )
        return ve0 + ve1

    def _predict_log_density(self, Fmu, Fvar, Y):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        Y_splits = [Y[..., 0], Y[..., 1]]
        nlpd0 = self.likelihoods[0]._predict_log_density(Fmu_splits[0], Fvar_splits[0], Y_splits[0])
        nlpd1 = self.likelihoods[1]._predict_log_density(Fmu_splits[1], Fvar_splits[1], Y_splits[1])
        return nlpd0 + nlpd1

    def _predict_log_density_single(self, Fmu, Fvar, Y, output):
        Fmu = self._split_fs(Fmu)[output]
        Fvar = self._split_fs(Fvar)[output]
        return self.likelihoods[output]._predict_log_density(Fmu, Fvar, Y)

    def _predict_mean_and_var(self, Fmu, Fvar):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        mean0, var0 = self.likelihoods[0].predict_mean_and_var(Fmu_splits[0], Fvar_splits[0])
        mean1, var1 = self.likelihoods[1].predict_mean_and_var(Fmu_splits[1], Fvar_splits[1])
        return tf.stack([mean0, mean1], axis=-1), tf.stack([var0, var1], axis=-1)
