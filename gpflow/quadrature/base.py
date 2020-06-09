from collections.abc import Iterable
import abc

import tensorflow as tf


class GaussianQuadrature:
    @abc.abstractmethod
    def _build_X_W(self, mean, var):
        raise NotImplementedError

    def logspace(self, fun, mean, var, *args, **kwargs):
        r"""
        Compute the Gaussian log-Expectation of a function f
            q(x) = N(mean, var)
            E_{X~q(x)}[f(X)] = log∫f(x)q(x)dx
        Using the approximation
            log \sum exp(f(x_i) + log w_i)
        Where x_i, w_i must be provided by some inheriting class
        """
        X, W = self._build_X_W(mean, var)
        logW = tf.math.log(W)
        if isinstance(fun, Iterable):
            return [tf.reduce_logsumexp(f(X, *args, **kwargs) + logW, axis=-2) for f in fun]
        return tf.reduce_logsumexp(fun(X, *args, **kwargs) + logW, axis=-2)

    def __call__(self, fun, mean, var, *args, **kwargs):
        r"""
        Compute the Gaussian Expectation of a function f
            q(x) = N(mean, var)
            E_{X~q(x)}[f(X)] = ∫f(x)q(x)dx
        Using the approximation
            \sum f(x_i)*w_i
        Where x_i, w_i must be provided by some inheriting class
        """
        X, W = self._build_X_W(mean, var)
        if isinstance(fun, Iterable):
            # Maybe this can be better implemented by stacking [f1(X), f2(X), ...]
            # and sum-reducing all at once
            # The problem: there is no garantee that f1(X), f2(X), ...
            # have comaptible shapes
            return [tf.reduce_sum(f(X, *args, **kwargs) * W, axis=-2) for f in fun]
        return tf.reduce_sum(fun(X, *args, **kwargs) * W, axis=-2)
