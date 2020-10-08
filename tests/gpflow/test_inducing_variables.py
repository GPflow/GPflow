import numpy as np
import tensorflow as tf

import gpflow


def test_inducing_points_with_None_shape():
    N, M1, D, P = 50, 13, 3, 1
    X, Y = np.random.randn(N, D), np.random.randn(N, P)

    Z1 = np.random.randn(M1, D)

    iv = gpflow.inducing_variables.InducingPoints(Z1)
    # overwrite Parameter with explicit tf.Variable with None shape:
    iv.Z = tf.Variable(Z1, trainable=False, dtype=gpflow.default_float(), shape=(None, D))

    m = gpflow.models.SGPR(data=(X, Y), kernel=gpflow.kernels.Matern32(), inducing_variable=iv)

    Z2 = np.random.randn(M1 + 1, D)
    m.inducing_variable.Z.assign(Z2)

    opt = tf.optimizers.Adam()

    @tf.function
    def optimization_step():
        opt.minimize(m.training_loss, m.trainable_variables)

    optimization_step()
