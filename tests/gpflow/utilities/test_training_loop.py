from typing import Any

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.experimental.check_shapes import ShapeChecker
from gpflow.utilities import training_loop


class Datum:
    rng = np.random.RandomState(123)
    cs = ShapeChecker().check_shape

    N = 13
    X = cs(np.linspace(0, 10, N)[:, None], "[N, 1]")
    noise_scale = 0.01
    Y = cs(np.sin(X) + rng.randn(N, 1) * noise_scale, "[N, 1]")
    data = (X, Y)


def create_model() -> gpflow.models.GPModel:
    return gpflow.models.GPR(
        Datum.data, gpflow.kernels.SquaredExponential(), noise_variance=Datum.noise_scale ** 2
    )


def assert_models_close(
    m: gpflow.models.GPModel, mref: gpflow.models.GPModel, **tol_kwargs: Any
) -> None:
    np.testing.assert_allclose(
        m.kernel.variance.numpy(), mref.kernel.variance.numpy(), **tol_kwargs
    )
    np.testing.assert_allclose(
        m.kernel.lengthscales.numpy(), mref.kernel.lengthscales.numpy(), **tol_kwargs
    )
    np.testing.assert_allclose(
        m.likelihood.variance.numpy(), mref.likelihood.variance.numpy(), **tol_kwargs
    )


def test_training_loop_compiles() -> None:
    m1 = create_model()
    m2 = create_model()
    training_loop(
        m1.training_loss, tf.optimizers.Adam(), m1.trainable_variables, maxiter=50, compile=True
    )
    training_loop(
        m2.training_loss, tf.optimizers.Adam(), m2.trainable_variables, maxiter=50, compile=False
    )
    assert_models_close(m1, m2)


def test_training_loop_converges() -> None:
    m = create_model()
    mref = create_model()
    gpflow.optimizers.Scipy().minimize(mref.training_loss, mref.trainable_variables)
    training_loop(
        m.training_loss,
        tf.optimizers.Adam(learning_rate=0.01),
        m.trainable_variables,
        maxiter=5000,
        compile=True,
    )
    assert_models_close(m, mref, rtol=1e-5)
