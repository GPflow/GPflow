from gpflow.likelihoods import Likelihood, Bernoulli, Poisson

from .utils import inv_probit
from .. import logdensities
import tensorflow as tf

class MultiStageLikelihood(Likelihood):

    def __init__(self, invlink_bernoulli=inv_probit,
                 invlink_poisson=tf.exp, **kwargs):
        super().__init__(**kwargs)
        self.invlink_bernoulli = invlink_bernoulli
        self.invlink_poisson = invlink_poisson

    def _split_f(self, F):
        F0 = F[..., 0:1]
        F1 = F[..., 1:2]
        F2 = F[..., 2:3]
        return (F0, F1, F2)

    def _log_prob(self, F, Y):
        F0, F1, F2 = self._split_f(F)
        return self._scalar_log_prob(F1, F2, Y)

    def _scalar_log_prob(self, F0, F1, F2, Y):
        raise NotImplementedError

    def _variational_expectations(self, Fmu, Fvar, Y):
        Fmu0, Fmu1, Fmu2 = self._split_f(Fmu)
        Fvar0, Fvar1, Fvar2 = self._split_f(Fvar)
        # flags
        Y0 = tf.cast(tf.equal(Y, 0), dtype=Y.dtype)
        Y1 = tf.cast(tf.equal(Y, 1), dtype=Y.dtype)
        Y2 = tf.cast(tf.greater_equal(Y, 2), dtype=Y.dtype)

        bern = Bernoulli(invlink=self.invlink_bernoulli)
        poisson = Poisson(invlink=self.invlink_poisson)
        ones = tf.ones_like(Y)
        zeros = tf.zeros_like(Y)

        ve0 = bern.variational_expectations(Fmu0, Fvar0, Y0)
        ven0 = bern.variational_expectations(-Fmu0, Fvar0, Y0)
        ve1 = bern.variational_expectations(Fmu1, Fvar1, Y1)
        ven1 = bern.variational_expectations(-Fmu1, Fvar1, Y1)
        ve2 = poisson.variational_expectations(Fmu2, Fvar2, Y)
        return Y0 * ve0 + \
            Y1 * (ven0 + ve1) + \
            Y2 * (ven0 + ven1 + ve2)

    def _predict_log_density(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def _predict_mean_and_var(self, Fmu, Fvar, Y):
        raise NotImplementedError

    def variational_expectations(self, Fmu, Fvar, Y):
        return self._variational_expectations(Fmu, Fvar, Y)
