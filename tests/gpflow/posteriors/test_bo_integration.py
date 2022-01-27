#  Copyright 2022 The GPflow Contributors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
from typing import Any, Callable, DefaultDict, Dict, Iterator, Mapping, Set, Tuple, Type, TypeVar

import numpy as np
import pytest
import tensorflow as tf
from _pytest.fixtures import SubRequest

import gpflow
from gpflow.base import RegressionData
from gpflow.config import default_float
from gpflow.inducing_variables import InducingPoints, InducingVariables
from gpflow.kernels import Kernel, Matern52
from gpflow.likelihoods import Exponential, Likelihood
from gpflow.models import GPR, SGPR, SVGP, VGP, GPModel, training_loss_closure
from gpflow.models.vgp import update_vgp_data
from gpflow.posteriors import AbstractPosterior, PrecomputeCacheType

_CreateModel = Callable[[RegressionData], GPModel]
_C = TypeVar("_C", bound=_CreateModel)

_MULTI_OUTPUT = "multi_output"
_MODEL_FACTORIES: Dict[_CreateModel, Mapping[str, Any]] = {}

# This exists to make it easy to disable tf.function, for debugging.
_COMPILE = True
_MAXITER = 500
_DEFAULT_ATOL = 1e-10
_DEFAULT_RTOL = 1e-7


@pytest.fixture(name="register_posterior_bo_integration_test")
def _register_posterior_bo_integration_test(
    request: SubRequest,
    tested_posteriors: DefaultDict[str, Set[Type[AbstractPosterior]]],
) -> Callable[[AbstractPosterior], None]:
    def _register_posterior(posterior: AbstractPosterior) -> None:
        tested_posteriors[request.function.__name__].add(posterior.__class__)

    return _register_posterior


def model_factory(
    *flags: str, atol: float = _DEFAULT_ATOL, rtol: float = _DEFAULT_RTOL
) -> Callable[[_C], _C]:
    """ Decorator for adding a function to the `_MODEL_FACTORIES` list. """

    properties = {
        "atol": atol,
        "rtol": rtol,
        **{flag: True for flag in flags},
    }

    def register(create_model: _C) -> _C:
        _MODEL_FACTORIES[create_model] = properties
        return create_model

    return register


def create_kernel() -> Kernel:
    return Matern52()


def create_likelihood() -> Likelihood:
    return Exponential()


def create_inducing_points(data: RegressionData) -> InducingPoints:
    n_features = data[0].shape[1]
    n_inducing_points = 25
    rng = np.random.default_rng(20220208)
    Z = tf.constant(rng.random((n_inducing_points, n_features)))
    return InducingPoints(Z)


def create_q(
    inducing_variable: InducingVariables, *, row_scale: int = 1, column_scale: int = 1
) -> Tuple[bool, tf.Tensor, tf.Tensor]:
    n_inducing_points = inducing_variable.num_inducing
    rng = np.random.default_rng(20220133)
    q_diag = True
    q_mu = tf.constant(rng.random((row_scale * n_inducing_points, column_scale)))
    q_sqrt = tf.constant(rng.random((row_scale * n_inducing_points, column_scale))) ** 2
    return q_diag, q_mu, q_sqrt


@model_factory(rtol=1e-3)
def create_gpr(data: RegressionData) -> GPR:
    return GPR(data=data, kernel=create_kernel())


@model_factory(rtol=1e-4)
def create_sgpr(data: RegressionData) -> SGPR:
    return SGPR(data=data, kernel=create_kernel(), inducing_variable=create_inducing_points(data))


@model_factory(rtol=1e-3)
def create_vgp(data: RegressionData) -> VGP:
    return VGP(data=data, kernel=create_kernel(), likelihood=create_likelihood())


@model_factory()
def create_svgp__independent_single_output(data: RegressionData) -> SVGP:
    inducing_variable = create_inducing_points(data)
    q_diag, q_mu, q_sqrt = create_q(inducing_variable)
    return SVGP(
        kernel=create_kernel(),
        likelihood=create_likelihood(),
        inducing_variable=inducing_variable,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )


@model_factory(_MULTI_OUTPUT)
def create_svgp__fully_correlated_multi_output(data: RegressionData) -> SVGP:
    n_outputs = data[1].shape[1]
    kernel = gpflow.kernels.SharedIndependent(create_kernel(), output_dim=n_outputs)
    inducing_variable = create_inducing_points(data)
    q_diag, q_mu, q_sqrt = create_q(inducing_variable, row_scale=n_outputs)
    return SVGP(
        kernel=kernel,
        likelihood=create_likelihood(),
        inducing_variable=inducing_variable,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )


@model_factory(_MULTI_OUTPUT)
def create_svgp__independent_multi_output(data: RegressionData) -> SVGP:
    n_outputs = data[1].shape[1]
    kernel = gpflow.kernels.SharedIndependent(create_kernel(), output_dim=n_outputs)
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        create_inducing_points(data)
    )
    q_diag, q_mu, q_sqrt = create_q(inducing_variable, column_scale=n_outputs)
    return SVGP(
        kernel=kernel,
        likelihood=create_likelihood(),
        inducing_variable=inducing_variable,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )


@model_factory(_MULTI_OUTPUT)
def create_svgp__fallback_independent_latent_posterior(data: RegressionData) -> SVGP:
    n_outputs = data[1].shape[1]
    rng = np.random.default_rng(20220131)
    kernel = gpflow.kernels.LinearCoregionalization(
        [create_kernel()],
        W=tf.constant(rng.standard_normal((n_outputs, 1))),
    )
    inducing_variable = gpflow.inducing_variables.FallbackSeparateIndependentInducingVariables(
        [create_inducing_points(data)]
    )
    q_diag, q_mu, q_sqrt = create_q(inducing_variable)
    return SVGP(
        kernel=kernel,
        likelihood=create_likelihood(),
        inducing_variable=inducing_variable,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )


@model_factory(_MULTI_OUTPUT)
def create_svgp__linear_coregionalization(data: RegressionData) -> SVGP:
    n_outputs = data[1].shape[1]
    rng = np.random.default_rng(20220131)
    kernel = gpflow.kernels.LinearCoregionalization(
        [create_kernel()], W=tf.constant(rng.standard_normal((n_outputs, 1)))
    )
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        create_inducing_points(data)
    )
    q_diag, q_mu, q_sqrt = create_q(inducing_variable)
    return SVGP(
        kernel=kernel,
        likelihood=create_likelihood(),
        inducing_variable=inducing_variable,
        q_diag=q_diag,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )


@pytest.fixture(params=_MODEL_FACTORIES)
def _create_model(request: SubRequest) -> _CreateModel:
    return request.param


@pytest.fixture
def _multi_output(_create_model: _CreateModel) -> bool:
    return _MULTI_OUTPUT in _MODEL_FACTORIES[_create_model]


@pytest.fixture
def _rtol(_create_model: _CreateModel) -> float:
    return _MODEL_FACTORIES[_create_model]["rtol"]


@pytest.fixture
def _atol(_create_model: _CreateModel) -> float:
    return _MODEL_FACTORIES[_create_model]["atol"]


@pytest.fixture
def _f_minimum(_multi_output: bool) -> tf.Tensor:
    return (
        tf.constant(
            [
                [0.2, 0.4],
                [0.4, 0.6],
                [0.6, 0.8],
            ],
            dtype=default_float(),
        )
        if _multi_output
        else tf.constant([[0.3, 0.5]], dtype=default_float())
    )


@pytest.fixture
def _f(_f_minimum: tf.Tensor) -> Callable[[tf.Tensor], tf.Tensor]:
    def f(X: tf.Tensor) -> tf.Tensor:
        err = X[:, None, :] - _f_minimum[None, :, :]
        err_sq = err ** 2
        return tf.reduce_sum(err_sq, axis=-1)

    return f


@pytest.fixture
def _data(
    _f: Callable[[tf.Tensor], tf.Tensor], _f_minimum: tf.Tensor
) -> Tuple[tf.Variable, tf.Variable]:
    n_initial_data = 10
    n_outputs, n_features = _f_minimum.shape

    rng = np.random.default_rng(20220126)
    X = tf.Variable(
        rng.random((n_initial_data, n_features)),
        shape=[None, n_features],
        dtype=default_float(),
        trainable=False,
    )
    Y = tf.Variable(
        _f(X),
        shape=[None, n_outputs],
        dtype=default_float(),
        trainable=False,
    )

    return X, Y


@pytest.fixture
def _extend_data(
    _data: Tuple[tf.Variable, tf.Variable], _f: Callable[[tf.Tensor], tf.Tensor]
) -> Callable[[GPModel], Iterator[int]]:
    n_iterations = 3
    rng = np.random.default_rng(20220127)
    X, Y = _data
    n_features = X.shape[1]

    def iterate(model: GPModel) -> Iterator[int]:
        for i in range(n_iterations):
            X_new = tf.constant(rng.random((1, n_features)))
            Y_new = _f(X_new)
            X_i = tf.concat([X, X_new], axis=0)
            Y_i = tf.concat([Y, Y_new], axis=0)

            if isinstance(model, VGP):
                update_vgp_data(model, (X_i, Y_i))
            else:
                X.assign(X_i)
                Y.assign(Y_i)
            yield i

    return iterate


@pytest.fixture
def _X_new(_data: Tuple[tf.Variable, tf.Variable]) -> tf.Tensor:
    rng = np.random.default_rng(20220128)
    X, _Y = _data
    n_features = X.shape[1]
    return tf.constant(rng.random((3, n_features)))


@pytest.fixture
def _optimize(_data: Tuple[tf.Variable, tf.Variable]) -> Callable[[GPModel], None]:
    def optimize(model: GPModel) -> None:
        gpflow.optimizers.Scipy().minimize(
            training_loss_closure(model, _data, compile=_COMPILE),
            variables=model.trainable_variables,
            options=dict(maxiter=_MAXITER),
            method="BFGS",
            compile=_COMPILE,
        )

    return optimize


def test_posterior_bo_integration__predict_f(
    register_posterior_bo_integration_test: Callable[[AbstractPosterior], None],
    _create_model: _CreateModel,
    _data: Tuple[tf.Variable, tf.Variable],
    _extend_data: Callable[[GPModel], Iterator[int]],
    _X_new: tf.Tensor,
    _rtol: float,
    _atol: float,
) -> None:
    """
    Check that data added incrementally is correctly reflected in `predict_f`.
    """
    _X, Y = _data
    n_rows_new = _X_new.shape[0]
    n_outputs = Y.shape[1]

    model = _create_model(_data)
    posterior = model.posterior(PrecomputeCacheType.VARIABLE)
    register_posterior_bo_integration_test(posterior)
    predict_f = posterior.predict_f
    if _COMPILE:
        predict_f = tf.function(predict_f)

    for _ in _extend_data(model):
        posterior.update_cache()
        compiled_mean, compiled_var = predict_f(_X_new)

        np.testing.assert_equal((n_rows_new, n_outputs), compiled_mean.shape)
        np.testing.assert_equal((n_rows_new, n_outputs), compiled_var.shape)

        eager_model = _create_model(_data)
        eager_mean, eager_var = eager_model.predict_f(_X_new)

        np.testing.assert_allclose(eager_mean, compiled_mean, rtol=_rtol, atol=_atol)
        np.testing.assert_allclose(eager_var, compiled_var, rtol=_rtol, atol=_atol)


def test_posterior_bo_integration__optimization(
    register_posterior_bo_integration_test: Callable[[AbstractPosterior], None],
    _create_model: _CreateModel,
    _data: Tuple[tf.Variable, tf.Variable],
    _extend_data: Callable[[GPModel], Iterator[int]],
    _X_new: tf.Tensor,
    _optimize: Callable[[GPModel], None],
    _rtol: float,
    _atol: float,
) -> None:
    """
    Check that data added incrementally is considered when optimizing a model.
    """
    _X, Y = _data
    n_rows_new = _X_new.shape[0]
    n_outputs = Y.shape[1]

    model = _create_model(_data)
    posterior = model.posterior(PrecomputeCacheType.VARIABLE)
    register_posterior_bo_integration_test(posterior)
    predict_f = posterior.predict_f
    if _COMPILE:
        predict_f = tf.function(predict_f)

    # Add all the data first, and then `optimize`, so that both models are optimized the same number
    # of times and with the same data, so they converge to the same result.

    for _ in _extend_data(model):
        pass

    _optimize(model)
    posterior.update_cache()
    compiled_mean, compiled_var = predict_f(_X_new)

    np.testing.assert_equal((n_rows_new, n_outputs), compiled_mean.shape)
    np.testing.assert_equal((n_rows_new, n_outputs), compiled_var.shape)

    eager_model = _create_model(_data)
    _optimize(eager_model)
    eager_mean, eager_var = eager_model.predict_f(_X_new)

    np.testing.assert_allclose(eager_mean, compiled_mean, rtol=_rtol, atol=_atol)
    np.testing.assert_allclose(eager_var, compiled_var, rtol=_rtol, atol=_atol)
