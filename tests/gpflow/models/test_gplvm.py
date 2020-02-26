# Copyright 2019 the GPflow authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional
import pytest

import numpy as np
import tensorflow as tf

import gpflow
from gpflow.utilities.ops import pca_reduce


class Data:
    rng = np.random.RandomState(999)
    N = 20
    D = 5
    Y = rng.randn(N, D)
    Q = 2
    M = 10
    X = rng.randn(N, Q)


def _run_optimize(kernel: Optional[tf.Module] = None):
    m = gpflow.models.GPLVM(Data.Y, Data.Q, kernel=kernel)
    log_likelihood_initial = m.log_likelihood()
    opt = gpflow.optimizers.Scipy()

    @tf.function
    def objective_closure():
        return - m.log_marginal_likelihood()

    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=2))
    assert m.log_likelihood() > log_likelihood_initial


def test_gplvm_default_kernel():
    _run_optimize()


def test_gplvm_periodic_kernel():
    kernel = gpflow.kernels.Periodic(base=gpflow.kernels.SquaredExponential())
    _run_optimize(kernel)


def test_bayesian_gplvm_1d():
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    m = gpflow.models.BayesianGPLVM(Data.Y,
                                    np.zeros((Data.N, Q)),
                                    np.ones((Data.N, Q)),
                                    kernel,
                                    inducing_variable=inducing_variable)
    assert len(m.inducing_variable) == Data.M
    log_likelihood_initial = m.log_likelihood()
    opt = gpflow.optimizers.Scipy()

    @tf.function
    def objective_closure():
        return - m.log_marginal_likelihood()

    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=2))
    assert m.log_likelihood() > log_likelihood_initial


def test_bayesian_gplvm_2d():
    Q = 2  # latent dimensions
    x_data_mean = pca_reduce(Data.Y, Q)
    kernel = gpflow.kernels.SquaredExponential()

    m = gpflow.models.BayesianGPLVM(Data.Y, x_data_mean, np.ones((Data.N, Q)), kernel, num_inducing_variables=Data.M)

    log_likelihood_initial = m.log_likelihood()
    opt = gpflow.optimizers.Scipy()

    @tf.function
    def objective_closure():
        return - m.log_marginal_likelihood()

    opt.minimize(objective_closure, m.trainable_variables, options=dict(maxiter=2))
    assert m.log_likelihood() > log_likelihood_initial

    # test prediction
    Xtest = Data.rng.randn(10, Q)
    mu_f, var_f = m.predict_f(Xtest)
    mu_fFull, var_fFull = m.predict_f(Xtest, full_cov=True)
    np.testing.assert_allclose(mu_fFull, mu_f)

    for i in range(Data.D):
        np.testing.assert_allclose(var_f[:, i], np.diag(var_fFull[:, :, i]))


def test_gplvm_constructor_checks():
    with pytest.raises(ValueError):
        assert Data.X.shape[1] == Data.Q
        latents_wrong_shape = Data.X[:, :Data.Q - 1]
        gpflow.models.GPLVM(Data.Y, Data.Q, x_data_mean=latents_wrong_shape)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, :Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, :Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q, x_data_mean=Data.X)

def test_bayesian_gplvm_constructor_check():
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    with pytest.raises(ValueError):
        gpflow.models.BayesianGPLVM(Data.Y,
                                    np.zeros((Data.N, Q)),
                                    np.ones((Data.N, Q)),
                                    kernel,
                                    inducing_variable=inducing_variable,
                                    num_inducing_variables=len(inducing_variable))
