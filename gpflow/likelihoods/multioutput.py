# Copyright 2020 The GPflow Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List

import tensorflow as tf

from .base import Likelihood


class MultiOutputLikelihood(Likelihood):
    def __init__(self, likelihoods: List[Likelihood]):
        assert all(l.observation_dim == 1 for l in likelihoods)

        self.likelihoods = likelihoods
        super().__init__(latent_dim=sum(self.latent_dims), observation_dim=len(likelihoods))

    @property
    def latent_dim(self):
        return sum([l.latent_dim for l in self.likelihoods])

    def _split_fs(self, F):
        Fs = []
        for idx, likelihood in enumerate(self.likelihoods):
            F_begin_idx = latent_function_cumsum
            F_end_idx = latent_function_cumsum + likelihood.latent_dim
            Fs.append(F[..., F_begin_idx:F_end_idx])
            latent_function_cumsum += likelihood.latent_dim
        return Fs

    def _log_prob(self, F, Y):
        Fs = self._split_fs(F)
        return tf.concat(
            [
                like.log_prob(Fi, Yi[..., None])
                for like, Fi, Yi in zip(self.likelihoods, Fs, tf.unstack(Y, axis=-1))
            ],
            axis=-1,
        )

    def _variational_expectations(self, Fmu, Fvar, Y):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        Y_splits = tf.unstack(Y, axis=-1)
        return tf.add_n(
            like.variational_expectations(Fmu_idx, Fvar_idx, Y_idx)
            for like, Fmu_idx, Fvar_idx, Y_idx in zip(
                self.likelihoods, Fmu_splits, Fvar_splits, Y_splits
            )
        )

    def _predict_log_density(self, Fmu, Fvar, Y):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        Y_splits = tf.unstack(Y, axis=-1)
        return tf.add_n(
            like.predict_log_density(Fmu_idx, Fvar_idx, Y_idx)
            for like, Fmu_idx, Fvar_idx, Y_idx in zip(
                self.likelihoods, Fmu_splits, Fvar_splits, Y_splits
            )
        )

    def _predict_mean_and_var(self, Fmu, Fvar):
        Fmu_splits = self._split_fs(Fmu)
        Fvar_splits = self._split_fs(Fvar)
        means, variances = [], []
        for like, Fmu_idx, Fvar_idx in zip(self.likelihoods, Fmu_splits, Fvar_splits):
            mean_idx, var_idx = like.predict_mean_and_var(Fmu_idx, Fvar_idx)
            means.append(mean_idx)
            variances.append(var_idx)

        return tf.stack(means, axis=-1), tf.stack(variances, axis=-1)
