# Copyright 2017 the GPflow authors.
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

import tensorflow as tf
import numpy as np

import gpflow
from gpflow import kernels
from gpflow.util import training_loop

import pytest

rng = np.random.RandomState(1)


class Data:
    N = 20  # number of data points
    D = 5  # data dimension
    Y = rng.randn(N, D)
    Q = 2  # latent dimensions
    M = 10  # inducing points


@pytest.mark.parametrize('kernel', [kernels.Periodic(), kernels.RBF()])
def test_gplvm_optimise_with_different_kernels(kernel):
    X_init = rng.rand(Data.N, Data.Q) if isinstance(kernel, kernels.Periodic) else None
    m = gpflow.models.GPLVM(Data.Y, Data.Q, X_mean=X_init, kernel=kernel)
    linit = m.log_likelihood()

    def closure():
        return m.neg_log_marginal_likelihood()

    training_loop(closure, optimizer=tf.optimizers.Adam(), var_list=m.trainable_variables,
                  maxiter=2)
    assert m.log_likelihood() > linit


@pytest.mark.parametrize('Q', [1, 2])
def test_bayesian_gplvm_1d_and_2d(Q):
    X_mean = np.zeros((Data.N, Q)) if Q == 1 else gpflow.models.PCA_reduce(Data.Y, Q)
    kernel = kernels.RBF() if Q == 1 else kernels.RBF(ard=False)
    if Q == 1:
        Z = np.expand_dims(np.linspace(0, 1, Data.M), Q)
    else:
        # By default we initialize by subset of initial latent points
        Z = np.random.permutation(X_mean)[:Data.M]  # inducing points
    m = gpflow.models.BayesianGPLVM(X_mean=X_mean,
                                    X_var=np.ones((Data.N, Q)),
                                    Y=Data.Y, kernel=kernel, feature=Z)
    linit = m.log_likelihood()

    def closure():
        return m.neg_log_marginal_likelihood()

    training_loop(closure, optimizer=tf.optimizers.Adam(), var_list=m.trainable_variables,
                  maxiter=2)
    assert m.log_likelihood() > linit

    if Q == 2:
        Xtest = rng.randn(10, Q)
        mu_f, var_f = m.predict_f(Xtest)
        mu_fFull, var_fFull = m.predict_f(Xtest, full_cov=True)
        assert np.allclose(mu_fFull, mu_f)
        # check full covariance diagonal
        for i in range(Data.D):
            assert np.allclose(var_f[:, i], np.diag(var_fFull[:, :, i]))


