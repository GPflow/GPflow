# %%
import numpy as np
import tensorflow as tf
import gpflow

tf.random.set_seed(99012)

# %%
class Datum:
    tolerance = 1e-06
    Yshape = (10, 3)
    Y = tf.random.normal(Yshape, dtype=tf.float64)
    F = tf.random.normal(Yshape, dtype=tf.float64)
    Fmu = tf.random.normal(Yshape, dtype=tf.float64)
    Fvar = 0.01 * tf.random.normal(Yshape, dtype=tf.float64) ** 2
    Fvar_zero = tf.zeros(Yshape, dtype=tf.float64)


class LikelihoodSetup(object):
    def __init__(self, likelihood, Y=Datum.Y, rtol=1e-06, atol=0.0):
        self.likelihood = likelihood
        self.Y = Y
        self.rtol = rtol
        self.atol = atol

    def __repr__(self):
        name = self.likelihood.__class__.__name__
        return f"{name}-rtol={self.rtol}-atol={self.atol}"


likelihood_setup = LikelihoodSetup(
    likelihood=gpflow.likelihoods.MultiClass(3), 
    Y=tf.argmax(Datum.Y, 1).numpy().reshape(-1, 1), 
    rtol=1e-3, atol=1e-3,
)

# %%
mu = Datum.Fmu
var = Datum.Fvar
likelihood = likelihood_setup.likelihood 
y = likelihood_setup.Y

F1 = likelihood.variational_expectations(mu, var, y)

def quadrature_log_prob(F, Y):
    return tf.expand_dims(likelihood.log_prob(F, Y), axis=-1)

for n_gh in (20, 21, 22, 23, 24, 25):
    pass
    quadrature = gpflow.quadrature.NDiagGHQuadrature(
        dim=3, n_gh=n_gh
    )
    F2 = tf.squeeze(quadrature(quadrature_log_prob, mu, var, Y=y), axis=-1)

    atol = np.max(np.abs((F1-F2).numpy()))
    rtol = np.max(np.abs((F1-F2).numpy())/np.abs(F1.numpy()))

    print(f'n_gh: {n_gh} - atol: {atol} - rtol: {rtol}')

# %%
np.testing.assert_allclose(F1, F2, rtol=likelihood_setup.rtol, atol=likelihood_setup.atol)
