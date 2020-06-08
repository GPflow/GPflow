from collections.abc import Iterable
import abc

import tensorflow as tf


class GaussianQuadrature():

    @abc.abstractmethod
    def _build_X_W(self, mean, var):
        raise NotImplementedError

    def logspace(self, fun, mean, var, *args, **kwargs):
        X, W = self._build_X_W(mean, var)
        logW = tf.math.log(W)
        if isinstance(fun, Iterable):
            return [tf.reduce_logsumexp(f(X, *args, **kwargs) + logW, axis=0) for f in fun]
        return tf.reduce_logsumexp(fun(X, *args, **kwargs) + logW, axis=0)

    def __call__(self, fun, mean, var, *args, **kwargs):
        X, W = self._build_X_W(mean, var)
        if isinstance(fun, Iterable):
            # Maybe this can be better implemented by concating [f1(X), f2(X), ...]
            # and sum-reducing all at once
            return [tf.reduce_sum(f(X, *args, **kwargs) * W, axis=0) for f in fun]
        return tf.reduce_sum(fun(X, *args, **kwargs) * W, axis=0)
