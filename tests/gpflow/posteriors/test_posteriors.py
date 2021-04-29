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
import warnings
from inspect import isabstract

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.ci_utils
from gpflow.conditionals import conditional
from gpflow.models.util import inducingpoint_wrapper
from gpflow.posteriors import (
    AbstractPosterior,
    FallbackIndependentLatentPosterior,
    FullyCorrelatedPosterior,
    IndependentPosteriorMultiOutput,
    IndependentPosteriorSingleOutput,
    LinearCoregionalizationPosterior,
    PrecomputeCacheType,
    create_posterior,
)

INPUT_DIMS = 2
NUM_INDUCING_POINTS = 3


@pytest.fixture(name="q_sqrt_factory", params=[0, 1, 2])
def _q_sqrt_factory_fixture(request):
    """
    When upgrading to Python 3.10, this can be replaced with a match-case statement.
    """
    if request.param == 0:

        def fn_0(_, __):
            return None

        return fn_0
    elif request.param == 1:

        def fn_1(n_inducing_points, num_latent_gps):
            # qsqrt: [M, L]
            return tf.random.normal((n_inducing_points, num_latent_gps), dtype=tf.float64) ** 2

        return fn_1
    elif request.param == 2:

        def fn_2(n_inducing_points, num_latent_gps):
            # qsqrt: [L, M, M]
            shape = (num_latent_gps, n_inducing_points, n_inducing_points)
            return tf.linalg.band_part(tf.random.normal(shape, dtype=tf.float64), -1, 0)

        return fn_2
    else:
        raise NotImplementedError


@pytest.fixture(name="whiten", params=[False, True])
def _whiten_fixture(request):
    return request.param


@pytest.fixture(name="num_latent_gps", params=[1, 2])
def _num_latent_gps_fixture(request):
    return request.param


@pytest.fixture(name="output_dims", params=[1, 5])
def _output_dims_fixture(request):
    return request.param


TESTED_POSTERIORS = set()


@pytest.fixture(scope="module", autouse=True)
def _ensure_all_posteriors_are_tested_fixture():
    """
    This fixture ensures that all concrete posteriors have unit tests which compare the predictions
    from the fused and precomputed code paths. When adding a new concrete posterior class to
    GPFlow, ensure that it is also tested in this manner.

    This autouse, module scoped fixture will always be executed when tests in this module are run.
    """
    # Code here will be executed before any of the tests in this module.

    yield  # Run tests in this module.

    # Code here will be executed after all of the tests in this module.

    available_posteriors = list(gpflow.ci_utils.subclasses(AbstractPosterior))
    concrete_posteriors = set([k for k in available_posteriors if not isabstract(k)])

    untested_posteriors = concrete_posteriors - TESTED_POSTERIORS

    if untested_posteriors:
        message = (
            f"No tests have been registered for the following posteriors: {untested_posteriors}."
        )
        if gpflow.ci_utils.is_continuous_integration():
            raise AssertionError(message)
        else:
            warnings.warn(message)


@pytest.fixture(name="register_posterior_test")
def _register_posterior_test_fixture():
    def _verify_and_register_posterior_test(posterior, expected_posterior_class):
        assert isinstance(posterior, expected_posterior_class)
        TESTED_POSTERIORS.add(expected_posterior_class)

    return _verify_and_register_posterior_test


def create_conditional(
    *, kernel, inducing_variable, q_mu, q_sqrt, whiten,
):
    def conditional_closure(Xnew, *, full_cov, full_output_cov):
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
    posterior, conditional_closure, full_cov, full_output_cov
):
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
    register_posterior_test, q_sqrt_factory, whiten, full_cov, full_output_cov
):
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, INPUT_DIMS))

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)
    q_sqrt = q_sqrt_factory(NUM_INDUCING_POINTS, 1)

    conditional = create_conditional(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorSingleOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fully_correlated_multi_output(
    register_posterior_test, q_sqrt_factory, full_cov, full_output_cov, whiten, output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, FullyCorrelatedPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_shk_shi(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_shk_sei(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_sek_shi(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_independent_multi_output_sek_sei(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, IndependentPosteriorMultiOutput)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fallback_independent_multi_output_sei(
    register_posterior_test, q_sqrt_factory, full_cov, full_output_cov, whiten, output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, FallbackIndependentLatentPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_fallback_independent_multi_output_shi(
    register_posterior_test, q_sqrt_factory, full_cov, full_output_cov, whiten, output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, FallbackIndependentLatentPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_linear_coregionalization_sei(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, LinearCoregionalizationPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


def test_linear_coregionalization_shi(
    register_posterior_test,
    q_sqrt_factory,
    full_cov,
    full_output_cov,
    whiten,
    num_latent_gps,
    output_dims,
):
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
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    posterior = create_posterior(
        kernel=kernel, inducing_variable=inducing_variable, q_mu=q_mu, q_sqrt=q_sqrt, whiten=whiten,
    )
    register_posterior_test(posterior, LinearCoregionalizationPosterior)

    _assert_fused_predict_f_equals_precomputed_predict_f_and_conditional(
        posterior, conditional, full_cov, full_output_cov
    )


@pytest.mark.parametrize(
    "precompute_cache_type", [PrecomputeCacheType.NOCACHE, PrecomputeCacheType.TENSOR]
)
def test_posterior_update_cache_with_variables_no_precompute(
    q_sqrt_factory, whiten, precompute_cache_type
):
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

    assert isinstance(posterior.alpha, tf.Variable)
    assert isinstance(posterior.Qinv, tf.Variable)


def test_posterior_update_cache_with_variables_update_value(q_sqrt_factory, whiten):
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
    initial_alpha = posterior.alpha
    initial_Qinv = posterior.Qinv

    posterior.update_cache(PrecomputeCacheType.VARIABLE)

    # ensure the values of alpha and Qinv will change
    q_mu.assign_add(tf.ones_like(q_mu))
    if initial_q_sqrt is not None:
        q_sqrt.assign_add(tf.ones_like(q_sqrt))
    posterior.update_cache(PrecomputeCacheType.VARIABLE)

    # assert that the values have changed
    assert not np.allclose(initial_alpha, tf.convert_to_tensor(posterior.alpha))
    if initial_q_sqrt is not None:
        assert not np.allclose(initial_Qinv, tf.convert_to_tensor(posterior.Qinv))


def test_posterior_update_cache_fails_without_argument(q_sqrt_factory, whiten):
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
    assert posterior.alpha is None
    assert posterior.Qinv is None

    with pytest.raises(ValueError):
        posterior.update_cache()

    posterior.update_cache(PrecomputeCacheType.TENSOR)
    assert isinstance(posterior.alpha, tf.Tensor)
    assert isinstance(posterior.Qinv, tf.Tensor)

    posterior.update_cache(PrecomputeCacheType.NOCACHE)
    assert posterior._precompute_cache == PrecomputeCacheType.NOCACHE
    assert posterior.alpha is None
    assert posterior.Qinv is None

    posterior.update_cache(PrecomputeCacheType.TENSOR)  # set posterior._precompute_cache
    assert posterior._precompute_cache == PrecomputeCacheType.TENSOR
    posterior.alpha = posterior.Qinv = None  # clear again

    posterior.update_cache()  # does not raise an exception
    assert isinstance(posterior.alpha, tf.Tensor)
    assert isinstance(posterior.Qinv, tf.Tensor)


def test_posterior_create_with_variables_update_cache_works(q_sqrt_factory, whiten):
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
    assert isinstance(posterior.alpha, tf.Variable)
    assert isinstance(posterior.Qinv, tf.Variable)

    alpha = posterior.alpha
    Qinv = posterior.Qinv

    posterior.update_cache()

    assert posterior.alpha is alpha
    assert posterior.Qinv is Qinv
