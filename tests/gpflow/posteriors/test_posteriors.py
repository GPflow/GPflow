#  Copyright 2021 The GPflow Contributors. All Rights Reserved.
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
from typing import Any, Callable, DefaultDict, Optional, Set, Type, cast

import numpy as np
import pytest
import tensorflow as tf
from _pytest.fixtures import SubRequest

import gpflow
import gpflow.ci_utils
from gpflow.base import TensorType
from gpflow.conditionals import conditional
from gpflow.experimental.check_shapes import check_shapes
from gpflow.inducing_variables import InducingPoints, InducingVariables
from gpflow.kernels import Kernel
from gpflow.likelihoods import Gaussian
from gpflow.mean_functions import Zero
from gpflow.models.util import inducingpoint_wrapper
from gpflow.posteriors import (
    AbstractPosterior,
    FallbackIndependentLatentPosterior,
    FullyCorrelatedPosterior,
    GPRPosterior,
    IndependentPosteriorMultiOutput,
    IndependentPosteriorSingleOutput,
    LinearCoregionalizationPosterior,
    PrecomputeCacheType,
    SGPRPosterior,
    VGPPosterior,
    create_posterior,
)

INPUT_DIMS = 2
NUM_INDUCING_POINTS = 3


# `PosteriorType` really should be something like `Type[AbstractPosterior]`, except mypy doesn't
# allow passing abstract classes to functions. See: https://github.com/python/mypy/issues/4717
PosteriorType = Type[Any]
RegisterPosterior = Callable[[AbstractPosterior, PosteriorType], None]


@pytest.fixture(name="register_posterior_test")
def _register_posterior_test_fixture(
    tested_posteriors: DefaultDict[str, Set[Type[AbstractPosterior]]]
) -> RegisterPosterior:
    def _verify_and_register_posterior_test(
        posterior: AbstractPosterior, expected_posterior_class: Type[AbstractPosterior]
    ) -> None:
        assert isinstance(posterior, expected_posterior_class)
        tested_posteriors["test_posteriors.py"].add(expected_posterior_class)

    return _verify_and_register_posterior_test


QSqrtFactory = Callable[[int, int], Optional[TensorType]]


@pytest.fixture(name="q_sqrt_factory", params=[0, 1, 2])
def _q_sqrt_factory_fixture(request: SubRequest) -> QSqrtFactory:
    """
    When upgrading to Python 3.10, this can be replaced with a match-case statement.
    """
    if request.param == 0:

        def fn_0(n_inducing_points: int, num_latent_gps: int) -> Optional[TensorType]:
            return None

        return fn_0
    elif request.param == 1:

        def fn_1(n_inducing_points: int, num_latent_gps: int) -> Optional[TensorType]:
            # qsqrt: [M, L]
            return tf.random.normal((n_inducing_points, num_latent_gps), dtype=tf.float64) ** 2

        return fn_1
    elif request.param == 2:

        def fn_2(n_inducing_points: int, num_latent_gps: int) -> Optional[TensorType]:
            # qsqrt: [L, M, M]
            shape = (num_latent_gps, n_inducing_points, n_inducing_points)
            return tf.linalg.band_part(tf.random.normal(shape, dtype=tf.float64), -1, 0)

        return fn_2
    else:
        raise NotImplementedError


@pytest.fixture(name="whiten", params=[False, True])
def _whiten_fixture(request: SubRequest) -> bool:
    return cast(bool, request.param)


@pytest.fixture(name="num_latent_gps", params=[1, 2])
def _num_latent_gps_fixture(request: SubRequest) -> int:
    return cast(int, request.param)


@pytest.fixture(name="output_dims", params=[1, 5])
def _output_dims_fixture(request: SubRequest) -> int:
    return cast(int, request.param)


ConditionalClosure = Callable[..., tf.Tensor]


@check_shapes(
    "inducing_variable: [M, D, broadcast P]",
    "q_mu: [MxP, R]",
    "q_sqrt: [MxP_or_MxP_N_N...]",
)
def create_conditional(
    *,
    kernel: Kernel,
    inducing_variable: InducingVariables,
    q_mu: TensorType,
    q_sqrt: TensorType,
    whiten: bool,
) -> ConditionalClosure:
    @check_shapes(
        "Xnew: [batch..., N, D]",
        "return[0]: [batch..., N, R]",
        "return[1]: [batch..., N, R] if (not full_cov) and (not full_output_cov)",
        "return[1]: [batch..., R, N, N] if full_cov and (not full_output_cov)",
        "return[1]: [batch..., N, R, R] if (not full_cov) and full_output_cov",
        "return[1]: [batch..., N, R, N, R] if full_cov and full_output_cov",
    )
    def conditional_closure(
        Xnew: TensorType, *, full_cov: bool, full_output_cov: bool
    ) -> tf.Tensor:
        return conditional(
            Xnew,
            inducing_variable,
            kernel,
            q_mu,
            q_sqrt=q_sqrt,
            white=whiten,
            full_cov=full_cov,
            full_output_cov=full_output_cov,
        )

    return conditional_closure


def _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
    posterior: AbstractPosterior,
    conditional_closure: ConditionalClosure,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    Xnew = np.random.randn(13, INPUT_DIMS)

    fused_f_mean, fused_f_cov = posterior.fused_predict_f(
        Xnew, full_cov=full_cov, full_output_cov=full_output_cov
    )

    precomputed_f_mean, precomputed_f_cov = posterior.predict_f(
        Xnew, full_cov=full_cov, full_output_cov=full_output_cov
    )

    conditional_f_mean, conditional_f_cov = conditional_closure(
        Xnew, full_cov=full_cov, full_output_cov=full_output_cov
    )

    np.testing.assert_allclose(fused_f_mean, precomputed_f_mean)
    np.testing.assert_allclose(fused_f_cov, precomputed_f_cov)
    np.testing.assert_array_equal(fused_f_mean, conditional_f_mean)
    np.testing.assert_array_equal(fused_f_cov, conditional_f_cov)


def test_independent_single_output(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    whiten: bool,
    full_cov: bool,
    full_output_cov: bool,
) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorSingleOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fully_correlated_multi_output(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    output_dims: int,
) -> None:
    """
    The fully correlated posterior has one latent GP.
    """
    kernel = gpflow.kernels.SharedIndependent(
        gpflow.kernels.SquaredExponential(), output_dim=output_dims
    )
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = np.random.randn(output_dims * NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(output_dims * NUM_INDUCING_POINTS, 1)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, FullyCorrelatedPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_shk_shi(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Independent multi-output posterior with a shared kernel and shared inducing points.
    """
    kernel = gpflow.kernels.SharedIndependent(
        gpflow.kernels.SquaredExponential(), output_dim=output_dims
    )
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_shk_sei(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Independent multi-output posterior with a shared kernel and separate inducing points.
    """
    kernel = gpflow.kernels.SharedIndependent(
        gpflow.kernels.SquaredExponential(), output_dim=output_dims
    )
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [
            inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
            for _ in range(num_latent_gps)
        ]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_sek_shi(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Independent multi-output posterior with separate independent kernels and shared inducing points.
    """
    kernel = gpflow.kernels.SeparateIndependent(
        [gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)]
    )
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_sek_sei(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Independent multi-output posterior with separate independent kernel and separate inducing points.
    """
    kernel = gpflow.kernels.SeparateIndependent(
        [gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)]
    )
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [
            inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
            for _ in range(num_latent_gps)
        ]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fallback_independent_multi_output_sei(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    output_dims: int,
) -> None:
    """
    Fallback posterior with separate independent inducing variables.

    The FallbackIndependentLatentPosterior is a subclass of the FullyCorrelatedPosterior which
    requires a single latent GP function.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential()], W=tf.random.normal((output_dims, 1))
    )
    inducing_variable = gpflow.inducing_variables.FallbackSeparateIndependentInducingVariables(
        [inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS)) for _ in range(1)]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, FallbackIndependentLatentPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fallback_independent_multi_output_shi(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    output_dims: int,
) -> None:
    """
    Fallback posterior with shared independent inducing variables.

    The FallbackIndependentLatentPosterior is a subclass of the FullyCorrelatedPosterior which
    requires a single latent GP function.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential()], W=tf.random.normal((output_dims, 1))
    )
    inducing_variable = gpflow.inducing_variables.FallbackSharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, FallbackIndependentLatentPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_linear_coregionalization_sei(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Linear coregionalization posterior with separate independent inducing variables.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)],
        W=tf.random.normal((output_dims, num_latent_gps)),
    )
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [
            inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
            for _ in range(num_latent_gps)
        ]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, LinearCoregionalizationPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_linear_coregionalization_shi(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    full_cov: bool,
    full_output_cov: bool,
    whiten: bool,
    num_latent_gps: int,
    output_dims: int,
) -> None:
    """
    Linear coregionalization with shared independent inducing variables.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)],
        W=tf.random.normal((output_dims, num_latent_gps)),
    )
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, num_latent_gps)

    conditional = create_conditional(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
    )
    register_posterior_test(posterior, LinearCoregionalizationPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


@pytest.mark.parametrize(
    "precompute_cache_type", [PrecomputeCacheType.NOCACHE, PrecomputeCacheType.TENSOR]
)
def test_posterior_update_cache_with_variables_no_precompute(
    q_sqrt_factory: QSqrtFactory, whiten: bool, precompute_cache_type: PrecomputeCacheType
) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    posterior = IndependentPosteriorSingleOutput(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        precompute_cache=precompute_cache_type,
    )
    posterior.update_cache(PrecomputeCacheType.VARIABLE)

    assert posterior.cache
    alpha, Qinv = posterior.cache
    assert isinstance(alpha, tf.Variable)
    assert isinstance(Qinv, tf.Variable)


@pytest.mark.parametrize(
    "precompute_cache_type", [PrecomputeCacheType.NOCACHE, PrecomputeCacheType.TENSOR]
)
def test_gpr_posterior_update_cache_with_variables_no_precompute(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    whiten: bool,
    precompute_cache_type: PrecomputeCacheType,
) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    X = np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS)
    Y = np.random.randn(NUM_INDUCING_POINTS, 1)

    posterior = GPRPosterior(
        kernel=kernel,
        data=(X, Y),
        likelihood=Gaussian(0.1),
        precompute_cache=precompute_cache_type,
        mean_function=Zero(),
    )
    posterior.update_cache(PrecomputeCacheType.VARIABLE)
    register_posterior_test(posterior, GPRPosterior)

    assert posterior.cache
    err, Lm = posterior.cache
    assert isinstance(err, tf.Variable)
    assert isinstance(Lm, tf.Variable)


@pytest.mark.parametrize(
    "precompute_cache_type", [PrecomputeCacheType.NOCACHE, PrecomputeCacheType.TENSOR]
)
def test_sgpr_posterior_update_cache_with_variables_no_precompute(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    whiten: bool,
    precompute_cache_type: PrecomputeCacheType,
) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    X = np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS)
    Y = np.random.randn(NUM_INDUCING_POINTS, 1)
    Z = np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS)

    posterior = SGPRPosterior(
        kernel=kernel,
        data=(X, Y),
        inducing_variable=InducingPoints(Z),
        likelihood=Gaussian(0.1),
        num_latent_gps=1,
        precompute_cache=precompute_cache_type,
        mean_function=Zero(),
    )
    posterior.update_cache(PrecomputeCacheType.VARIABLE)
    register_posterior_test(posterior, SGPRPosterior)

    assert posterior.cache
    L, LB, c = posterior.cache
    assert isinstance(L, tf.Variable)
    assert isinstance(LB, tf.Variable)
    assert isinstance(c, tf.Variable)


@pytest.mark.parametrize(
    "precompute_cache_type", [PrecomputeCacheType.NOCACHE, PrecomputeCacheType.TENSOR]
)
def test_vgp_posterior_update_cache_with_variables_no_precompute(
    register_posterior_test: RegisterPosterior,
    q_sqrt_factory: QSqrtFactory,
    whiten: bool,
    precompute_cache_type: PrecomputeCacheType,
) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    X = np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS)
    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    posterior = VGPPosterior(
        kernel=kernel,
        X=X,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        mean_function=Zero(),
        precompute_cache=precompute_cache_type,
    )
    posterior.update_cache(PrecomputeCacheType.VARIABLE)
    register_posterior_test(posterior, VGPPosterior)

    assert posterior.cache
    (Lm,) = posterior.cache
    assert isinstance(Lm, tf.Variable)


def test_posterior_update_cache_with_variables_update_value(
    q_sqrt_factory: QSqrtFactory, whiten: bool
) -> None:
    # setup posterior
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = tf.Variable(np.random.randn(NUM_INDUCING_POINTS, 1))

    initial_q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)
    if initial_q_sqrt is not None:
        q_sqrt = tf.Variable(initial_q_sqrt)
    else:
        q_sqrt = initial_q_sqrt

    posterior = IndependentPosteriorSingleOutput(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        precompute_cache=PrecomputeCacheType.TENSOR,
    )
    assert posterior.cache
    initial_alpha, initial_Qinv = posterior.cache

    posterior.update_cache(PrecomputeCacheType.VARIABLE)

    # ensure the values of alpha and Qinv will change
    q_mu.assign_add(tf.ones_like(q_mu))
    if initial_q_sqrt is not None:
        q_sqrt.assign_add(tf.ones_like(q_sqrt))
    posterior.update_cache(PrecomputeCacheType.VARIABLE)

    # assert that the values have changed
    assert posterior.cache
    alpha, Qinv = posterior.cache
    assert not np.allclose(initial_alpha, tf.convert_to_tensor(alpha))
    if initial_q_sqrt is not None:
        assert not np.allclose(initial_Qinv, tf.convert_to_tensor(Qinv))


def test_posterior_update_cache_fails_without_argument(
    q_sqrt_factory: QSqrtFactory, whiten: bool
) -> None:
    # setup posterior
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = tf.Variable(np.random.randn(NUM_INDUCING_POINTS, 1))

    initial_q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)
    if initial_q_sqrt is not None:
        q_sqrt = tf.Variable(initial_q_sqrt)
    else:
        q_sqrt = initial_q_sqrt

    posterior = IndependentPosteriorSingleOutput(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        precompute_cache=None,
    )
    assert posterior.cache is None

    with pytest.raises(ValueError):
        posterior.update_cache()

    posterior.update_cache(PrecomputeCacheType.TENSOR)
    assert posterior.cache
    assert all(isinstance(c, tf.Tensor) for c in posterior.cache)

    posterior.update_cache(PrecomputeCacheType.NOCACHE)
    assert posterior._precompute_cache == PrecomputeCacheType.NOCACHE
    assert posterior.cache is None

    posterior.update_cache(PrecomputeCacheType.TENSOR)  # set posterior._precompute_cache
    assert posterior._precompute_cache == PrecomputeCacheType.TENSOR

    posterior.cache = None  # clear again
    posterior.update_cache()  # does not raise an exception
    assert posterior.cache
    assert all(isinstance(c, tf.Tensor) for c in posterior.cache)


def test_posterior_create_with_variables_update_cache_works(
    q_sqrt_factory: QSqrtFactory, whiten: bool
) -> None:
    # setup posterior
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = tf.Variable(np.random.randn(NUM_INDUCING_POINTS, 1))

    initial_q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)
    if initial_q_sqrt is not None:
        q_sqrt = tf.Variable(initial_q_sqrt)
    else:
        q_sqrt = initial_q_sqrt

    posterior = IndependentPosteriorSingleOutput(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        precompute_cache=PrecomputeCacheType.VARIABLE,
    )
    assert posterior.cache
    assert all(isinstance(c, tf.Variable) for c in posterior.cache)

    cache = posterior.cache

    posterior.update_cache()

    assert posterior.cache is cache
