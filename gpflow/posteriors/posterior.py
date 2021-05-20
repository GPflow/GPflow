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
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Tuple, cast

import tensorflow as tf

from .. import mean_functions
from ..base import Module
from ..types import MeanAndVariance


class PrecomputeCacheType(Enum):
    """
    - `PrecomputeCacheType.TENSOR` (or `"tensor"`): Precomputes the cached
      quantities and stores them as tensors (which allows differentiating
      through the prediction). This is the default.
    - `PrecomputeCacheType.VARIABLE` (or `"variable"`): Precomputes the cached
      quantities and stores them as variables, which allows for updating
      their values without changing the compute graph (relevant for AOT
      compilation).
    - `PrecomputeCacheType.NOCACHE` (or `"nocache"` or `None`): Avoids
      immediate cache computation. This is useful for avoiding extraneous
      computations when you only want to call the posterior's
      `fused_predict_f` method.
    """

    TENSOR = "tensor"
    VARIABLE = "variable"
    NOCACHE = "nocache"


class Posterior(Module, ABC):
    def __init__(
        self,
        kernel,
        mean_function: Optional[mean_functions.MeanFunction] = None,
        *,
        precompute_cache: Optional[PrecomputeCacheType],
    ):
        """
        Users should use `create_posterior` to create instances of concrete
        subclasses of this Posterior class instead of calling this constructor
        directly. For `create_posterior` to be able to correctly instantiate
        subclasses, developers need to ensure their subclasses don't change the
        constructor signature.
        """
        self.kernel = kernel
        self.mean_function = mean_function

        self.alpha = self.Qinv = None
        if precompute_cache is not None:
            self.update_cache(precompute_cache)

    def update_cache(self, precompute_cache: Optional[PrecomputeCacheType] = None):
        """
        Sets the cache depending on the value of `precompute_cache` to a
        `tf.Tensor`, `tf.Variable`, or clears the cache. If `precompute_cache`
        is not given, the setting defaults to the most-recently-used one.
        """
        if precompute_cache is None:
            try:
                precompute_cache = cast(
                    PrecomputeCacheType, self._precompute_cache,  # type: ignore
                )
            except AttributeError:
                raise ValueError(
                    "You must pass precompute_cache explicitly (the cache had not been updated before)."
                )
        else:
            self._precompute_cache = precompute_cache

        if precompute_cache is PrecomputeCacheType.NOCACHE:
            self.alpha = self.Qinv = None

        elif precompute_cache is PrecomputeCacheType.TENSOR:
            self.alpha, self.Qinv = self._precompute()

        elif precompute_cache is PrecomputeCacheType.VARIABLE:
            alpha, Qinv = self._precompute()
            if isinstance(self.alpha, tf.Variable) and isinstance(self.Qinv, tf.Variable):
                # re-use existing variables
                self.alpha.assign(alpha)
                self.Qinv.assign(Qinv)
            else:  # create variables
                self.alpha = tf.Variable(alpha, trainable=False)
                self.Qinv = tf.Variable(Qinv, trainable=False)

    def _add_mean_function(self, Xnew, mean):
        if self.mean_function is None:
            return mean
        else:
            return mean + self.mean_function(Xnew)

    def fused_predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function
        Does not make use of caching
        """
        mean, cov = self._conditional_fused(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self._add_mean_function(Xnew, mean), cov

    @abstractmethod
    def _conditional_fused(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function
        Does not make use of caching
        """

    def predict_f(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, including mean_function.
        Relies on precomputed alpha and Qinv (see _precompute method)
        """
        if self.alpha is None or self.Qinv is None:
            raise ValueError(
                "Cache has not been precomputed yet. Call update_cache first or use fused_predict_f"
            )
        mean, cov = self._conditional_with_precompute(
            Xnew, full_cov=full_cov, full_output_cov=full_output_cov
        )
        return self._add_mean_function(Xnew, mean), cov

    @abstractmethod
    def _precompute(self) -> Tuple[tf.Tensor, tf.Tensor]:
        """
        Precomputes alpha and Qinv that do not depend on Xnew
        """

    @abstractmethod
    def _conditional_with_precompute(
        self, Xnew, full_cov: bool = False, full_output_cov: bool = False
    ) -> MeanAndVariance:
        """
        Computes predictive mean and (co)variance at Xnew, *excluding* mean_function.
        Relies on precomputed alpha and Qinv (see _precompute method)
        """


class _QDistribution(Module):
    """
    Base class for our parametrization of q(u) in the `VariationalPosteriorMixin`.
    Internal - do not rely on this outside of GPflow.
    """


class _DeltaDist(_QDistribution):
    def __init__(self, q_mu):
        self.q_mu = q_mu  # [M, L]

    @property
    def q_sqrt(self):
        return None


class _DiagNormal(_QDistribution):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [M, L]


class _MvNormal(_QDistribution):
    def __init__(self, q_mu, q_sqrt):
        self.q_mu = q_mu  # [M, L]
        self.q_sqrt = q_sqrt  # [L, M, M], lower-triangular


class VariationalPosteriorMixin:

    _q_dist: _QDistribution

    @property
    def q_mu(self):
        return self._q_dist.q_mu

    @property
    def q_sqrt(self):
        return self._q_dist.q_sqrt

    def _set_qdist(self, q_mu, q_sqrt):
        if q_sqrt is None:
            self._q_dist = _DeltaDist(q_mu)
        elif len(q_sqrt.shape) == 2:  # q_diag
            self._q_dist = _DiagNormal(q_mu, q_sqrt)
        else:
            self._q_dist = _MvNormal(q_mu, q_sqrt)
