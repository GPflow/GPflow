# %%
import numpy as np
import tensorflow as tf
import gpflow

# %%
np.random.seed(1)  # for reproducibility


def generate_data(N=80):
    X = np.random.rand(N)[:, None] * 10 - 5  # Inputs, shape N x 1
    F = 2.5 * np.sin(6 * X) + np.cos(3 * X)  # Mean function values
    NoiseVar = 2 * np.exp(-((X - 2) ** 2) / 4) + 0.3  # Noise variances
    Y = F + np.random.randn(N, 1) * np.sqrt(NoiseVar)  # Noisy data
    return X, Y, NoiseVar


X, Y, NoiseVar = generate_data()
Y_data = np.hstack([Y, NoiseVar])


# %%
class HeteroskedasticGaussian(gpflow.likelihoods.QuadratureLikelihood):
    def __init__(self, **kwargs):
        # this likelihood expects a single latent function F, and two columns in the data matrix Y:
        super().__init__(latent_dim=1, observation_dim=2, **kwargs)

    def _log_prob(self, F, Y):
        # log_prob is used by the quadrature fallback of variational_expectations and predict_log_density.
        this_Y = tf.expand_dims(Y[:, 0], axis=-1)
        NoiseVar = tf.expand_dims(Y[:, 1], axis=-1)
        result = gpflow.logdensities.gaussian(this_Y, F, NoiseVar)

        # Squeezing is needed to pass the test likelihood._check_return_shape
        result = tf.squeeze(result, axis=-1)

        # Commented below is the old implementation.
        # It is not working, as it returns result with shape [80, 80] instead of [80]
        # this_Y, NoiseVar = Y[:, 0], Y[:, 1]
        # result = gpflow.logdensities.gaussian(this_Y, F, NoiseVar)

        return result

    def analytical_variational_expectations(self, Fmu, Fvar, Y):
        Y, NoiseVar = Y[:, 0], Y[:, 1]
        Fmu, Fvar = Fmu[:, 0], Fvar[:, 0]
        return (
            -0.5 * np.log(2 * np.pi)
            - 0.5 * tf.math.log(NoiseVar)
            - 0.5 * (tf.math.square(Y - Fmu) + Fvar) / NoiseVar
        )


# %% Model
likelihood = HeteroskedasticGaussian()
kernel = gpflow.kernels.Matern52(lengthscales=0.5)
model = gpflow.models.VGP((X, Y_data), kernel=kernel, likelihood=likelihood, num_latent_gps=1)

# %% Test log_prob
F = model.predict_f_samples(X)
log_prob = likelihood.log_prob(F, Y_data)


# %% Test variational_expectations
# Will call the quadrature code from inside
Fmu, Fvar = model.predict_f(X)
var_exp = likelihood.variational_expectations(Fmu, Fvar, Y_data)
analytical_var_exp = likelihood.analytical_variational_expectations(Fmu, Fvar, Y_data)
np.testing.assert_allclose(var_exp, analytical_var_exp)


# %%
