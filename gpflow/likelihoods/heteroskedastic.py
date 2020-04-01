import numpy as np
import tensorflow as tf

from ..quadrature import ndiagquad
from .likelihoods import Likelihood


class HeteroskedasticLikelihood(Likelihood):
    def __init__(self, observation_dim: int = 1):
        if observation_dim != 1:
            raise NotImplementedError(
                "HeteroskedasticLikelihood currently assumes a single output dimension"
            )
        latent_dim = 2 * observation_dim
        super().__init__(latent_dim=latent_dim, observation_dim=observation_dim)
        self.num_gauss_hermite_points = 20

    def _split_f(self, F):
        F1 = F[..., 0:1]
        F2 = F[..., 1:2]
        return (F1, F2)

    def _log_prob(self, F, Y):
        F1, F2 = self._split_f(F)
        return self._scalar_log_prob(F1, F2, Y)

    @abc.abstractmethod
    def _scalar_log_prob(self, F1, F2, Y):
        raise NotImplementedError

    def _scalar_conditional_mean(self, F1, F2):
        raise NotImplementedError

    def _scalar_conditional_variance(self, F1, F2):
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
            self._scalar_log_prob, self.num_gauss_hermite_points, Fmu_tuple, Fvar_tuple, Y=Y
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

        def integrand(*X):
            return self._scalar_conditional_mean(*X) + self._scalar_conditional_variance(*X) ** 2

        integrands = [self._scalar_conditional_mean, integrand]
        E_y, E_y2 = ndiagquad(integrands, self.num_gauss_hermite_points, Fmu_tuple, Fvar_tuple)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y
