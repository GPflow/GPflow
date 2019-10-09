from collections.abc import Iterable
from functools import reduce

import tensorflow as tf

from .base import Combination
from ..base import Parameter, positive


class ChangePoints(Combination):
    r"""
    The ChangePoints kernel defines a fixed number of change-points along a 1d
    input space where different kernels govern different parts of the space.

    The kernel is by multiplication and addition of the base kernels with
    sigmoid functions (σ). A single change-point kernel is defined as:

        K₁(x, x') * (1 - σ(x)) * (1 - σ(x')) + K₂(x, x') * σ(x) * σ(x')

    where K₁ is deactivated around the change-point and K₂ is activated. The
    single change-point version can be found in \citet{lloyd2014}. Each sigmoid
    is a logistic function defined as:

        σ(x) = 1 / (1 + exp{-s(x - x₀)})

    parameterized by location "x₀" and steepness "s".

    @incollection{lloyd2014,
      author = {Lloyd, James Robert et al},
      title = {Automatic Construction and Natural-language Description of Nonparametric Regression Models},
      booktitle = {Proceedings of the Twenty-Eighth AAAI Conference on Artificial Intelligence},
      year = {2014},
      url = {http://dl.acm.org/citation.cfm?id=2893873.2894066},
    }
    """
    def __init__(self, kernels, locations, steepness=1.0, name=None):
        """
        TODO: Describe params
        """
        if len(kernels) != len(locations) + 1:
            raise ValueError("Number of active kernels ({a}) must be one more than the number of "
                             "change-points ({cp})".format(a=len(kernels), cp=len(locations)))

        if isinstance(steepness, Iterable) and len(steepness) != len(locations):
            raise ValueError("Dimension of steepness ({ns}) does not match number of locations "
                             "({nl})".format(ns=len(steepness), nl=len(locations)))

        super().__init__(kernels, name=name)

        self.num_changepoints = len(locations)
        self.locations = Parameter(locations)
        self.steepness = Parameter(steepness, transform=positive())

    def _set_kernels(self, kernels):
        # it is not clear how to flatten out nested change-points
        self.kernels = list(kernels)

    def K(self, X, X2=None, presliced=False):
        sig_X = self._sigmoids(X)
        sig_X2 = self._sigmoids(X2) if X2 is not None else sig_X

        # `starters` are the sigmoids going from 0 -> 1, whilst `stoppers` go from 1 -> 0
        starters = sig_X * tf.transpose(sig_X2, perm=(1, 0, 2))
        stoppers = (1 - sig_X) * tf.transpose((1 - sig_X2), perm=(1, 0, 2))

        regime_Ks = []
        for i, k in enumerate(self.kernels):
            K = k.K(X, X2)
            if i > 0:
                K *= starters[:, :, i-1]
            if i < self.num_changepoints:
                K *= stoppers[:, :, i]
            regime_Ks.append(K)

        return reduce(tf.add, regime_Ks)

    def K_diag(self, X, presliced=False):
        return tf.matrix_diag_part(self.K(X))

    def _sigmoids(self, X):
        locations = tf.sort(self.locations)  # ensure locations are ordered
        locations = tf.reshape(locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        return tf.sigmoid(steepness * (X - locations))
