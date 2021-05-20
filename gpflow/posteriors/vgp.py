#  Copyright 2021 The GPflow Contributors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Tuple

import tensorflow as tf

from ..conditionals import conditional
from ..types import MeanAndVariance
from .posterior import Posterior, VariationalPosteriorMixin


class VariationalPosterior(Posterior, VariationalPosteriorMixin):
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        X_data, _ = self.data
        mu, var = conditional(
            Xnew, X_data, self.kernel, self.q_mu, q_sqrt=self.q_sqrt, full_cov=full_cov, white=True,
        )
        return self._add_mean_function(Xnew, mu), var

    def _precompute(self) -> Tuple[tf.Tensor, tf.Tensor]:
        pass

    def _conditional_with_precompute(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        pass
