import gpflow
import numpy as np
import tensorflow as tf
import pytest


def test_log_prior_with_no_prior():
    """
    A parameter without any prior should have zero log-prior,
    even if it has a transform to constrain it.
    """
    param = gpflow.Parameter(5.3, transform=gpflow.positive())
    assert param.log_prior().numpy() == 0.0


np.random.seed(1)

class Datum:
    X = 10 * np.random.randn(5,1)
    Y = 10 * np.random.randn(5,1)

def test_gpr_objective_equivalence():
    data = (Datum.X, Datum.Y)
    l_value = 1.3
    l_variable = tf.Variable(l_value, dtype=gpflow.default_float(), trainable=True)
    m1 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential(lengthscale=l_value))
    m2 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential(lengthscale=l_variable))
    m2.kernel.lengthscale._transform = None
    assert np.allclose(m1.neg_log_marginal_likelihood().numpy(),
                       m2.neg_log_marginal_likelihood().numpy())
