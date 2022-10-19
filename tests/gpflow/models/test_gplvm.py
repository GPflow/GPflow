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

from typing import AbstractSet, Any, Callable, Optional

import numpy as np
import pytest
import tensorflow as tf

import gpflow
from gpflow.utilities.ops import pca_reduce


@pytest.mark.parametrize(
    "X,expect_change",
    [
        ([], False),
        ([1], False),
        (range(5), True),
        (
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
                [7.0, 8.0],
            ],
            True,
        ),
    ],
)
def test_stateless_shuffle(X: Any, expect_change: bool, mk_seed: Callable[[], tf.Tensor]) -> None:
    def _to_tuple(a: Any) -> Any:
        try:
            return tuple(_to_tuple(aa) for aa in a)
        except TypeError:
            return a

    def _to_set(a: tf.Tensor) -> AbstractSet[Any]:
        return set(_to_tuple(a.numpy()))

    s1 = mk_seed()
    shuffle = gpflow.models.gplvm._stateless_shuffle
    tf_X = tf.constant(X)
    v11 = shuffle(tf_X, s1)
    assert v11.shape == tf_X.shape
    assert len(tf_X) == len(v11)
    assert _to_set(tf_X) == _to_set(v11)
    v12 = shuffle(tf_X, s1)

    assert (v11 == v12).numpy().all()

    if expect_change:
        s2 = mk_seed()
        v2 = shuffle(tf_X, s2)
        assert not (v11 == v2).numpy().all()


class Data:
    rng = np.random.RandomState(999)
    N = 20
    D = 5
    Y = rng.randn(N, D)
    Q = 2
    M = 10
    X = rng.randn(N, Q)


@pytest.mark.parametrize(
    "kernel",
    [
        None,  # default kernel: SquaredExponential
        gpflow.kernels.Periodic(base_kernel=gpflow.kernels.SquaredExponential()),
    ],
)
def test_gplvm_with_kernels(kernel: Optional[gpflow.kernels.Kernel], seed: tf.Tensor) -> None:
    m = gpflow.models.GPLVM(Data.Y, Data.Q, kernel=kernel)
    lml_initial = m.log_marginal_likelihood()
    opt = gpflow.optimizers.Scipy()
    opt.minimize(m.training_loss_closure(seed=seed), m.trainable_variables, options=dict(maxiter=2))
    assert m.log_marginal_likelihood() > lml_initial


def test_bayesian_gplvm_1d(mk_seed: Callable[[], tf.Tensor]) -> None:
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    m = gpflow.models.BayesianGPLVM(
        Data.Y,
        np.zeros((Data.N, Q)),
        np.ones((Data.N, Q)),
        kernel,
        inducing_variable=inducing_variable,
        seed=mk_seed(),
    )
    assert m.inducing_variable.num_inducing == Data.M

    elbo_initial = m.elbo(seed=mk_seed())
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        m.training_loss_closure(seed=mk_seed()), m.trainable_variables, options=dict(maxiter=2)
    )
    assert m.elbo(seed=mk_seed()) > elbo_initial


def test_bayesian_gplvm_2d(mk_seed: Callable[[], tf.Tensor]) -> None:
    Q = 2  # latent dimensions
    X_data_mean = pca_reduce(Data.Y, Q)
    kernel = gpflow.kernels.SquaredExponential()

    m = gpflow.models.BayesianGPLVM(
        Data.Y,
        X_data_mean,
        np.ones((Data.N, Q)),
        kernel,
        num_inducing_variables=Data.M,
        seed=mk_seed(),
    )

    elbo_initial = m.elbo(seed=mk_seed())
    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        m.training_loss_closure(seed=mk_seed()), m.trainable_variables, options=dict(maxiter=2)
    )
    assert m.elbo(seed=mk_seed()) > elbo_initial

    # test prediction
    Xtest = Data.rng.randn(10, Q)
    mu_f, var_f = m.predict_f(Xtest)
    mu_fFull, var_fFull = m.predict_f(Xtest, full_cov=True)
    np.testing.assert_allclose(mu_fFull, mu_f)

    for i in range(Data.D):
        np.testing.assert_allclose(var_f[:, i], np.diag(var_fFull[i, :, :]))


def test_gplvm_constructor_checks() -> None:
    with pytest.raises(ValueError):
        assert Data.X.shape[1] == Data.Q
        latents_wrong_shape = Data.X[:, : Data.Q - 1]
        gpflow.models.GPLVM(Data.Y, Data.Q, X_data_mean=latents_wrong_shape)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, : Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q)
    with pytest.raises(ValueError):
        observations_wrong_shape = Data.Y[:, : Data.Q - 1]
        gpflow.models.GPLVM(observations_wrong_shape, Data.Q, X_data_mean=Data.X)


def test_bayesian_gplvm_constructor_check(seed: tf.Tensor) -> None:
    Q = 1
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = np.linspace(0, 1, Data.M)[:, None]
    with pytest.raises(ValueError):
        gpflow.models.BayesianGPLVM(
            Data.Y,
            np.zeros((Data.N, Q)),
            np.ones((Data.N, Q)),
            kernel,
            inducing_variable=inducing_variable,
            num_inducing_variables=len(inducing_variable),
            seed=seed,
        )
