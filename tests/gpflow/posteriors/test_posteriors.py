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

from dataclasses import dataclass
from inspect import isabstract
from itertools import chain

import numpy as np
import pytest
import tensorflow as tf

import gpflow
import gpflow.ci_utils
from gpflow.models.util import inducingpoint_wrapper
from gpflow.posterior import (
    AbstractPosterior,
    FallbackIndependentLatentPosterior,
    FullyCorrelatedPosterior,
    IndependentPosteriorMultiOutput,
    IndependentPosteriorSingleOutput,
    LinearCoregionalizationPosterior, get_posterior_class, create_posterior,
)

NUM_INDUCING_POINTS = 3


@pytest.fixture(name="mean_function", params=[None, 1.0])
def _mean_function_fixture(request):
    mean_function_parameter = request.param

    if mean_function_parameter is None:
        return None
    else:
        return gpflow.mean_functions.Constant(mean_function_parameter)


@pytest.fixture(name="set_q_sqrt", params=[False, True])
def _set_q_sqrt(request):
    return request.param


@pytest.fixture(name="whiten", params=[False, True])
def _whiten_fixture(request):
    return request.param


@pytest.fixture(name="precompute", params=[False, True])
def _precompute_fixture(request):
    return request.param


@pytest.fixture(name="input_dims", params=[1, 2])
def _input_dims_fixture(request):
    return request.param


@pytest.fixture(name="num_latent_gps", params=[1, 5])
def _num_latent_gps_fixture(request):
    return request.param


@pytest.fixture(name="output_dims", params=[1, 7])
def _output_dims_fixture(request):
    return request.param


_independent_single_output = [IndependentPosteriorSingleOutput]
_fully_correlated_multi_output = [FullyCorrelatedPosterior]
_independent_multi_output = [IndependentPosteriorMultiOutput]
_fallback_independent_multi_output = [FallbackIndependentLatentPosterior]
_linear_coregionalization = [LinearCoregionalizationPosterior]


def test_no_missing_kernels():
    tested_kernels = set(
        chain(
            _independent_single_output,
            _fully_correlated_multi_output,
            _independent_multi_output,
            _fallback_independent_multi_output,
            _linear_coregionalization,
        )
    )

    available_kernels = list(gpflow.ci_utils.subclasses(AbstractPosterior))
    concrete_kernels = set([k for k in available_kernels if not isabstract(k)])

    assert tested_kernels == concrete_kernels


def _assert_fused_predict_f_equals_precomputed_predict_f(
    posterior, precompute, full_cov, full_output_cov, input_dims, decimals=6
):
    Xnew = np.random.randn(13, input_dims)

    fused_f_mean, fused_f_cov = posterior.fused_predict_f(
        Xnew, full_cov=full_cov, full_output_cov=full_output_cov
    )

    if not precompute:
        posterior.update_cache()

    precomputed_f_mean, precomputed_f_cov = posterior.predict_f(
        Xnew, full_cov=full_cov, full_output_cov=full_output_cov
    )

    np.testing.assert_array_almost_equal(fused_f_mean, precomputed_f_mean, decimal=decimals)
    np.testing.assert_array_almost_equal(fused_f_cov, precomputed_f_cov, decimal=decimals)


@pytest.mark.parametrize("posterior_class", _independent_single_output)
def test_independent_single_output(
    posterior_class, set_q_sqrt, whiten, mean_function, precompute, full_cov, full_output_cov, input_dims
):
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims))

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.constant((np.random.randn(NUM_INDUCING_POINTS, 1) ** 2) * 0.01)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims
    )


@dataclass(frozen=True)
class LatentVariationalMultiOutputParameters:
    qsqrt = np.eye(NUM_INDUCING_POINTS)[None, :]
    qmean = np.random.randn(NUM_INDUCING_POINTS, 1)


@pytest.mark.parametrize("posterior_class", _fully_correlated_multi_output)
@pytest.mark.parametrize("set_q_sqrt", [True, False])
@pytest.mark.parametrize(
    "whiten",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="FullyCorrelatedPosterior and subclasses are not consistent between the "
                "fused and precomputed implementations with whiten=False."
            ),
        ),
    ],
)
def test_fully_correlated_multi_output(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, output_dims
):
    """
    The fully correlated posterior has one latent GP.
    """
    kernel = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential(), output_dim=output_dims)
    inducing_variable = inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims))

    q_mu = np.random.randn(output_dims * NUM_INDUCING_POINTS, 1)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(output_dims * NUM_INDUCING_POINTS, batch_shape=[1], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims
    )


@pytest.mark.parametrize("posterior_class", _independent_multi_output)
def test_independent_multi_output_shk_shi(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, num_latent_gps, output_dims
):
    """
    Independent multi-output posterior with a shared kernel and shared inducing points.
    """
    kernel = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential(), output_dim=output_dims)
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[num_latent_gps], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims, decimals=4
    )


@pytest.mark.parametrize("posterior_class", _independent_multi_output)
def test_independent_multi_output_shk_sei(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, num_latent_gps, output_dims
):
    """
    Independent multi-output posterior with a shared kernel and separate inducing points.
    """
    kernel = gpflow.kernels.SharedIndependent(gpflow.kernels.SquaredExponential(), output_dim=output_dims)
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims)) for _ in range(num_latent_gps)]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[num_latent_gps], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims, decimals=4
    )


@pytest.mark.parametrize("posterior_class", _independent_multi_output)
def test_independent_multi_output_sek_shi(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, num_latent_gps, output_dims
):
    """
    Independent multi-output posterior with separate independent kernels and shared inducing points.
    """
    kernel = gpflow.kernels.SeparateIndependent([gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)])
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[num_latent_gps], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims, decimals=3
    )


@pytest.mark.parametrize("posterior_class", _independent_multi_output)
def test_independent_multi_output_sek_sei(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, num_latent_gps, output_dims
):
    """
    Independent multi-output posterior with separate independent kernel and separate inducing points.
    """
    kernel = gpflow.kernels.SeparateIndependent([gpflow.kernels.SquaredExponential() for _ in range(num_latent_gps)])
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims)) for _ in range(num_latent_gps)]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, num_latent_gps)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[num_latent_gps], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims, decimals=4
    )


@pytest.mark.parametrize("posterior_class", _fallback_independent_multi_output)
@pytest.mark.parametrize(
    "whiten",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="FullyCorrelatedPosterior and subclasses are not consistent between the "
                "fused and precomputed implementations with whiten=False."
            ),
        ),
    ],
)
def test_fallback_independent_multi_output_sei(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, output_dims
):
    """
    Fallback posterior with separate independent inducing variables.

    The FallbackIndependentLatentPosterior is a subclass of the FullyCorrelatedPosterior which
    requires a single latent GP function.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential() for _ in range(1)], W=tf.ones((output_dims, 1))
    )
    inducing_variable = gpflow.inducing_variables.FallbackSeparateIndependentInducingVariables(
        [inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims)) for _ in range(1)]
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[1], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims
    )


@pytest.mark.parametrize("posterior_class", _fallback_independent_multi_output)
@pytest.mark.parametrize(
    "whiten",
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.xfail(
                reason="FullyCorrelatedPosterior and subclasses are not consistent between the "
                "fused and precomputed implementations with whiten=False."
            ),
        ),
    ],
)
def test_fallback_independent_multi_output_shi(
    posterior_class, set_q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten, input_dims, output_dims
):
    """
    Fallback posterior with shared independent inducing variables.

    The FallbackIndependentLatentPosterior is a subclass of the FullyCorrelatedPosterior which
    requires a single latent GP function.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential() for _ in range(1)], W=tf.ones((output_dims, 1))
    )
    inducing_variable = gpflow.inducing_variables.FallbackSharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, input_dims))
    )

    q_mu = np.random.randn(NUM_INDUCING_POINTS, 1)

    q_sqrt = None
    if set_q_sqrt:
        q_sqrt = tf.eye(NUM_INDUCING_POINTS, batch_shape=[1], dtype=tf.float64)

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, input_dims
    )


@pytest.mark.parametrize("posterior_class", _linear_coregionalization)
@pytest.mark.parametrize(
    "q_sqrt", [None, tf.constant(LatentVariationalMultiOutputParameters.qsqrt)]
)
def test_linear_coregionalization_sei(
    posterior_class, q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten
):
    """
    Linear coregionalization posterior with separate independent inducing variables.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential()], W=tf.ones((1, 1))
    )
    inducing_variable = gpflow.inducing_variables.SeparateIndependentInducingVariables(
        [inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, 1))]
    )

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=LatentVariationalMultiOutputParameters.qmean,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, decimals=4
    )


@pytest.mark.parametrize("posterior_class", _linear_coregionalization)
@pytest.mark.parametrize(
    "q_sqrt", [None, tf.constant(LatentVariationalMultiOutputParameters.qsqrt)]
)
def test_linear_coregionalization_shi(
    posterior_class, q_sqrt, mean_function, precompute, full_cov, full_output_cov, whiten
):
    """
    Linear coregionalization with shared independent inducing variables.
    """
    kernel = gpflow.kernels.LinearCoregionalization(
        [gpflow.kernels.SquaredExponential()], W=tf.ones((1, 1))
    )
    inducing_variable = gpflow.inducing_variables.SharedIndependentInducingVariables(
        inducingpoint_wrapper(np.random.randn(NUM_INDUCING_POINTS, 1))
    )

    posterior = posterior_class(
        kernel=kernel,
        inducing_variable=inducing_variable,
        q_mu=LatentVariationalMultiOutputParameters.qmean,
        q_sqrt=q_sqrt,
        whiten=whiten,
        mean_function=mean_function,
        precompute=precompute,
    )

    _assert_fused_predict_f_equals_precomputed_predict_f(
        posterior, precompute, full_cov, full_output_cov, decimals=4
    )
