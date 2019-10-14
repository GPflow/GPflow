import gpflow
import numpy as np
import tensorflow as tf
import pytest


np.random.seed(1)

class Datum:
    X = 10 * np.random.randn(5,1)
    Y = 10 * np.random.randn(5,1)
    lengthscale = 3.3


def test_gpr_objective_equivalence():
    """
    In Maximum Likelihood Estimation (MLE), i.e. when there are no priors on
    the parameters, the objective should not depend on any transforms on the
    parameters.
    We use GPR as a simple model that has an objective.
    """
    data = (Datum.X, Datum.Y)
    l_value = Datum.lengthscale

    l_variable = tf.Variable(l_value, dtype=gpflow.default_float(), trainable=True)
    m1 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential(lengthscale=l_value))
    m2 = gpflow.models.GPR(data, kernel=gpflow.kernels.SquaredExponential())
    m2.kernel.lengthscale = gpflow.Parameter(l_variable, transform=None)
    assert np.allclose(m1.kernel.lengthscale.numpy(),
                       m2.kernel.lengthscale.numpy())  # consistency check

    assert np.allclose(m1.neg_log_marginal_likelihood().numpy(),
                       m2.neg_log_marginal_likelihood().numpy()), \
            "MLE objective should not depend on Parameter transform"


def test_log_prior_with_no_prior():
    """
    A parameter without any prior should have zero log-prior,
    even if it has a transform to constrain it.
    """
    param = gpflow.Parameter(5.3, transform=gpflow.positive())
    assert param.log_prior().numpy() == 0.0
