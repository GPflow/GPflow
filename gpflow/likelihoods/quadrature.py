import tensorflow as tf

from .base import Likelihood
from ..quadrature import NDiagGHQuadrature


class QuadratureLikelihood(Likelihood):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._quadrature = None

    @property
    def quadrature(self):
        raise NotImplementedError()

    def _predict_mean_and_var(self, Fmu, Fvar):
        r"""
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :returns: mean and variance, both with shape [..., observation_dim]
        """
        def conditional_y_squared(F):
            return self.conditional_variance(F) + self.conditional_mean(F) ** 2
        integrands = [self.conditional_mean, conditional_y_squared]
        E_y, E_y2 = self.quadrature(integrands, Fmu, Fvar)
        V_y = E_y2 - E_y ** 2
        return E_y, V_y

    def _quadrature_log_prob(self, F, Y):
        return tf.expand_dims(self.log_prob(F, tf.expand_dims(Y, -2)), -1)

    def _predict_log_density(self, Fmu, Fvar, Y):
        r"""
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: variational expectations, with shape [...]
        """
        return tf.squeeze(self.quadrature.logspace(self._quadrature_log_prob, Fmu, Fvar, Y), -1)

    def _variational_expectations(self, Fmu, Fvar, Y):
        r"""
        :param Fmu: mean function evaluation Tensor, with shape [..., latent_dim]
        :param Fvar: variance of function evaluation Tensor, with shape [..., latent_dim]
        :param Y: observation Tensor, with shape [..., observation_dim]:
        :returns: log predictive density, with shape [...]
        """
        return tf.squeeze(self.quadrature(self._quadrature_log_prob, Fmu, Fvar, Y), -1)


class NDiagGHQuadratureLikelihood(QuadratureLikelihood):

    def __init__(self, latent_dim: int, n_gh: int = 20, **kwargs):
        super().__init__(latent_dim=latent_dim, **kwargs)
        self.quadrature_initialized = tf.Variable(False, trainable=False)
        self.n_gh = n_gh

    @property
    def quadrature(self):
        if not self.quadrature_initialized:
            if self.latent_dim is None:
                raise Exception(
                    'latent_dim not specified. '
                    'Either set likelihood.latent_dim directly or '
                    'call a method which passes data to have it inferred.'
                )
            self._quadrature = NDiagGHQuadrature(self.latent_dim, self.n_gh)
            self.quadrature_initialized.assign(True)
        return self._quadrature
