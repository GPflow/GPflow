import abc
from typing import Callable, List, Type

import tensorflow as tf
import tensorflow_probability as tfp

from ..quadrature import ndiagquad
from ..utilities import positive
from .base import Likelihood


# NOTE- in the following we're assuming outputs are independent, i.e. full_output_cov=False


class Heteroskedastic(Likelihood):
    r"""
    Heteroskedastic likelihood for which GPs parametrize not just location (mean) of
    the likelihood distribution but also other parameters (e.g. scale or variance).
    """

    def __init__(self, latent_dim: int, *, num_gauss_hermite_points: int = 21):
        super().__init__(latent_dim=latent_dim, observation_dim=1)
        self.num_gauss_hermite_points = num_gauss_hermite_points

    @staticmethod
    def _split_f(F):
        return tf.unstack(F, axis=-1)

    def _log_prob(self, F, Y):
        Fs = self._split_f(F)
        return self._scalar_log_prob(*Fs, Y=Y[..., 0])

    @abc.abstractmethod
    def _scalar_log_prob(self, *Fs, Y):
        raise NotImplementedError

    def _scalar_conditional_mean(self, *Fs):
        raise NotImplementedError

    def _scalar_conditional_variance(self, *Fs):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]:
        :returns: variational expectations, with shape [...]
        """
        Fmu_tuple = self._split_f(Fmu)
        Fvar_tuple = self._split_f(Fvar)
        return ndiagquad(
            self._scalar_log_prob, self.num_gauss_hermite_points, Fmu_tuple, Fvar_tuple, Y=Y,
        )

    def _predict_log_density(self, Fmu, Fvar, Y):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (Gaussian, Poisson) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., latent_dim]:
        :returns: log predictive density, with shape [...]
        """
        Fmu_tuple = self._split_f(Fmu)
        Fvar_tuple = self._split_f(Fvar)
        return ndiagquad(
            self._scalar_log_prob,
            self.num_gauss_hermite_points,
            Fmu_tuple,
            Fvar_tuple,
            logspace=True,
            Y=Y,
        )

    def _predict_mean_and_var(self, Fmu, Fvar):
        r"""
        Here, we implement a default Gauss-Hermite quadrature routine, but some
        likelihoods (e.g. Gaussian) will implement specific cases.
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance, both with shape [..., observation_dim]
        """
        Fmu_tuple = self._split_f(Fmu)
        Fvar_tuple = self._split_f(Fvar)

        def conditional_y_squared(*X):
            return self._scalar_conditional_variance(*X) + self._scalar_conditional_mean(*X) ** 2

        E_y, E_y2 = ndiagquad(
            [self._scalar_conditional_mean, conditional_y_squared],
            self.num_gauss_hermite_points,
            Fmu_tuple,
            Fvar_tuple,
        )
        V_y = E_y2 - E_y ** 2
        return tf.expand_dims(E_y, -1), tf.expand_dims(V_y, -1)


class HeteroskedasticTFPBase(Heteroskedastic):
    @abc.abstractmethod
    def _get_conditional_distribution(self, *Fs):
        raise NotImplementedError

    def _scalar_log_prob(self, *Fs, Y) -> tf.Tensor:
        """
        Computes the log pdf for `Y` under the specified distribution
        when the loc and the pre-transformed scale are given by `F1` and `F2`
        :param F1: [..., 1]
        :param F2: [..., 1]
        :param Y: [..., 1]
        :return: log pdf vector [...]
        """
        return self._get_conditional_distribution(*Fs).log_prob(Y)

    def _scalar_conditional_mean(self, *Fs):
        """ E_y[ y ] for y ~ p(y | f)"""
        return self._get_conditional_distribution(*Fs).mean()

    def _scalar_conditional_variance(self, *Fs):
        """ Var_y[ y ] for y ~ p(y | f)"""
        return self._get_conditional_distribution(*Fs).variance()


class HeteroskedasticTFPConditional(HeteroskedasticTFPBase):
    """
    Heteroskedastic likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    The `loc` and `scale` of the distribution are given by a
    two-dimensional multi-output GP.
    """

    def __init__(
        self,
        latent_dim: int,
        conditional_distribution: Callable[..., tfp.distributions.Distribution],
        **kwargs,
    ):
        """
        :param latent_dim: number of arguments to the `conditional_distribution` callable
        :param conditional_distribution: function from (F1, F2, ..., Fn) to a tfp Distribution,
            where n = `latent_dim`
            as first and second argument, respectivily.
        """
        super().__init__(latent_dim, **kwargs)
        self.conditional_distribution = conditional_distribution

    def _get_conditional_distribution(self, *Fs):
        return self.conditional_distribution(*Fs)


class HeteroskedasticTFPDistribution(HeteroskedasticTFPBase):
    # NOTE: this could instead subclass HeteroskedasticTFPConditional and pass
    # self._get_conditional_distribution as the conditional_distribution
    # argument, or None (as _get_conditional_distribution is overwritten), or
    # call HeteroskedasticTFPBase.__init__ instead...

    """
    Heteroskedastic Likelihood where the conditional distribution
    is given by a TensorFlow Probability Distribution.
    The `loc` and `scale` of the distribution are given by a
    two-dimensional multi-output GP.
    """

    def __init__(
        self,
        distribution_class: Type[tfp.distributions.Distribution],
        transform: tfp.bijectors.Bijector = positive(base="exp"),
        **kwargs,
    ):
        """
        :param distribution_class: distribution class parameterized by `loc` and `scale`
            as first and second argument, respectivily.
        :param transform: callable applied to the variance GP to make it positive.
            Typically, `tf.exp` or `tf.softplus`, but can be any function f: R -> R^+.
        """
        super().__init__(latent_dim=2, **kwargs)
        self.distribution_class = distribution_class
        self.transform = transform

    def _get_conditional_distribution(self, F1, F2) -> tfp.distributions.Distribution:
        loc = F1
        scale = self.transform.forward(F2)
        return self.distribution_class(loc, scale)

    def _scalar_log_prob(self, F1, F2, Y) -> tf.Tensor:
        """
        Computes the log pdf for `Y` under the specified distribution
        when the loc and the pre-transformed scale are given by `F1` and `F2`
        :param F1: [..., 1]
        :param F2: [..., 1]
        :param Y: [..., 1]
        :return: log pdf vector [...]
        """
        dists = self._get_conditional_distribution(F1, F2)
        return dists.log_prob(Y)  # [..., 1]

    def _scalar_conditional_mean(self, F1, F2):
        """ E_y[ y ] for y ~ p(y | f)"""
        return F1

    def _scalar_conditional_variance(self, F1, F2):
        """ Var_y[ y ] for y ~ p(y | f)"""
        return self.transform.forward(F2) ** 2
