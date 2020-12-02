import numpy as np
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

#class DiagNormal:
#    def __init__(self, q_mu, q_diag):
#        self.q_mu = gpflow.Parameter(q_mu)
#        self.q_diag = q_diag
#
#class MvnNormal:
#    def __init__(self, q_mu, q_sqrt):
#        self.q_mu = q_mu
#        self.q_sqrt = q_sqrt


class InducingPointsVariable(gpflow.inducing_variables.InducingVariables):
    def __init__(self, Z, q_dist: tfp.distributions.Distribution, whiten=True):
        self.Z = gpflow.Parameter(Z)
        self.q_dist = q_dist
        self.whiten = whiten

    def _cache(self, kernel):
        Kuu = covariances.Kuu(self, kernel)
        L = tf.linalg.cholesky(Kuu)
        if not self.whiten:
            # alpha = Kuu⁻¹ q_mu
            alpha = tf.linalg.cholesky_solve(L, tf.linalg.adjoint(self.q_dist.mean()))
        else:
            # alpha = L⁻T q_mu
            alpha = tf.linalg.triangular_solve(L, tf.linalg.adjoint(self.q_dist.mean()))
        # predictive variance = Kff - Kfu Qinv Kuf
        # S = q_sqrt q_sqrtT
        if not self.whiten:
            # Qinv = Kuu⁻¹ - Kuu⁻¹ S Kuu⁻¹
            #      = Kuu⁻¹ - L⁻T L⁻¹ S L⁻T L⁻¹
            #      = L⁻T (I - L⁻¹ S L⁻T) L⁻¹
            Linv_qsqrt = solve(L, q_sqrt)
            Linv_Sigma_LinvT = Linv_qsqrt @ Linv_qsqrt.T
            B = I - Linv_Sigma_LinvT
            Qinv = solve(L, solve(L, B) ... transposed)
        else:
            # Qinv = Kuu⁻¹ - L⁻T S L⁻¹
            # Linv = (L⁻¹ I) = solve(L, I)
            # Kinv = Linv.T @ Linv
            Kuu_inv = tf.linalg.inv(Kuu)

def conditional(

