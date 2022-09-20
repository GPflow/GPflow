# Copyright 2016 the GPflow authors.
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

from typing import Tuple, cast

import numpy as np
import pytest
from numpy.testing import assert_allclose

import gpflow
from gpflow.base import AnyNDArray
from gpflow.experimental.check_shapes import check_shapes
from gpflow.kernels import SquaredExponential
from gpflow.likelihoods import Gaussian
from tests.gpflow.kernels.reference import ref_rbf_kernel

rng = np.random.RandomState(1)

# ------------------------------------------
# Helpers
# ------------------------------------------


@check_shapes(
    "y: []",
    "K: []",
    "noise_var: []",
    "return: []",
)
def univariate_log_marginal_likelihood(
    y: AnyNDArray, K: AnyNDArray, noise_var: AnyNDArray
) -> AnyNDArray:
    return (  # type: ignore[no-any-return]
        -0.5 * y * y / (K + noise_var) - 0.5 * np.log(K + noise_var) - 0.5 * np.log(np.pi * 2.0)
    )


@check_shapes(
    "y: []",
    "K: []",
    "noise_var: []",
    "return[0]: []",
    "return[1]: []",
)
def univariate_posterior(
    y: AnyNDArray, K: AnyNDArray, noise_var: AnyNDArray
) -> Tuple[AnyNDArray, AnyNDArray]:
    mean = K * y / (K + noise_var)
    variance: AnyNDArray = K - K / (K + noise_var)
    return mean, variance


@check_shapes(
    "meanA: []",
    "meanB: []",
    "varA: []",
    "varB: []",
    "return: []",
)
def univariate_prior_KL(
    meanA: AnyNDArray, meanB: AnyNDArray, varA: AnyNDArray, varB: AnyNDArray
) -> AnyNDArray:
    # KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and qB are univariate normal distributions.
    return cast(
        AnyNDArray,
        0.5
        * (
            np.log(varB)
            - np.log(varA)
            - 1.0
            + varA / varB
            + cast(AnyNDArray, meanB - meanA) * cast(AnyNDArray, meanB - meanA) / varB
        ),
    )


@check_shapes(
    "meanA: [N, 1]",
    "covA: [N, N]",
    "meanB: [N, 1]",
    "covB: [N, N]",
    "return: [1, 1]",
)
def multivariate_prior_KL(
    meanA: AnyNDArray, covA: AnyNDArray, meanB: AnyNDArray, covB: AnyNDArray
) -> AnyNDArray:
    # KL[ qA | qB ] = E_{qA} \log [qA / qB] where qA and aB are
    # K dimensional multivariate normal distributions.
    # Analytically tractable and equal to...
    # 0.5 * (Tr(covB^{-1} covA) + (meanB - meanA)^T covB^{-1} (meanB - meanA)
    #        - K + log(det(covB)) - log (det(covA)))
    K = covA.shape[0]
    traceTerm = 0.5 * np.trace(np.linalg.solve(covB, covA))
    delta: AnyNDArray = meanB - meanA
    mahalanobisTerm = 0.5 * np.dot(delta.T, np.linalg.solve(covB, delta))
    constantTerm = -0.5 * K
    priorLogDeterminantTerm = 0.5 * np.linalg.slogdet(covB)[1]
    variationalLogDeterminantTerm = -0.5 * np.linalg.slogdet(covA)[1]
    return cast(
        AnyNDArray,
        (
            traceTerm
            + mahalanobisTerm
            + constantTerm
            + priorLogDeterminantTerm
            + variationalLogDeterminantTerm
        ),
    )


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Datum:
    num_latent_gps = 1
    y_data: AnyNDArray = np.array(2.0)
    X: AnyNDArray = np.atleast_2d(np.array([0.0]))
    Y: AnyNDArray = np.atleast_2d(np.array([y_data]))
    Z = X.copy()
    zero_mean: AnyNDArray = np.array(0.0)
    K: AnyNDArray = np.array(1.0)
    noise_var: AnyNDArray = np.array(0.5)
    posterior_mean, posterior_var = univariate_posterior(y=y_data, K=K, noise_var=noise_var)
    posterior_std = np.sqrt(posterior_var)
    data = (X, Y)


class MultiDatum:
    dim = 3
    num_latent_gps = 1
    Y = rng.randn(dim, 1)
    X = rng.randn(dim, 1)
    Z = X.copy()
    noise_var = 0.5
    signal_var: AnyNDArray = np.array(1.5)
    ls: AnyNDArray = np.array(1.7)
    q_mean = rng.randn(dim, num_latent_gps)
    q_sqrt_diag = rng.rand(dim, num_latent_gps)
    q_sqrt_full: AnyNDArray = np.tril(rng.rand(dim, dim))


def test_reference_implementation_consistency() -> None:
    q_mean = rng.rand(1, 1)
    q_cov = rng.rand(1, 1)
    p_mean = rng.rand(1, 1)
    p_cov = rng.rand(1, 1)

    multivariate_KL = multivariate_prior_KL(q_mean, p_mean, q_cov, p_cov)
    univariate_KL = univariate_prior_KL(
        q_mean.squeeze(), p_mean.squeeze(), q_cov.squeeze(), p_cov.squeeze()
    )

    assert_allclose(univariate_KL, multivariate_KL.squeeze(), atol=4)


@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("whiten", [True, False])
def test_variational_univariate_prior_KL(diag: bool, whiten: bool) -> None:
    reference_kl = univariate_prior_KL(
        Datum.posterior_mean, Datum.zero_mean, Datum.posterior_var, Datum.K
    )
    q_mu: AnyNDArray = np.ones((1, Datum.num_latent_gps)) * Datum.posterior_mean
    ones = np.ones((1, Datum.num_latent_gps)) if diag else np.ones((1, 1, Datum.num_latent_gps))
    q_sqrt = ones * Datum.posterior_std
    model = gpflow.models.SVGP(
        kernel=SquaredExponential(variance=Datum.K),
        likelihood=Gaussian(),
        inducing_variable=Datum.Z,
        num_latent_gps=Datum.num_latent_gps,
        q_diag=diag,
        whiten=whiten,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )
    assert_allclose(model.prior_kl(), reference_kl, atol=4)


@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("whiten", [True, False])
def test_variational_univariate_log_likelihood(diag: bool, whiten: bool) -> None:
    # reference marginal likelihood estimate
    reference_log_marginal_likelihood = univariate_log_marginal_likelihood(
        y=Datum.y_data, K=Datum.K, noise_var=Datum.noise_var
    )
    q_mu: AnyNDArray = np.ones((1, Datum.num_latent_gps)) * Datum.posterior_mean
    ones = np.ones((1, Datum.num_latent_gps)) if diag else np.ones((1, 1, Datum.num_latent_gps))
    q_sqrt = ones * Datum.posterior_std
    model = gpflow.models.SVGP(
        kernel=SquaredExponential(variance=Datum.K),
        likelihood=Gaussian(),
        inducing_variable=Datum.Z,
        num_latent_gps=Datum.num_latent_gps,
        q_diag=diag,
        whiten=whiten,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )
    model_likelihood = model.elbo(Datum.data).numpy()
    assert_allclose(model_likelihood, reference_log_marginal_likelihood, atol=4)


@pytest.mark.parametrize("diag", [True, False])
@pytest.mark.parametrize("whiten", [True, False])
def test_variational_univariate_conditionals(diag: bool, whiten: bool) -> None:
    q_mu: AnyNDArray = np.ones((1, Datum.num_latent_gps)) * Datum.posterior_mean
    ones = np.ones((1, Datum.num_latent_gps)) if diag else np.ones((1, 1, Datum.num_latent_gps))
    q_sqrt = ones * Datum.posterior_std
    model = gpflow.models.SVGP(
        kernel=SquaredExponential(variance=Datum.K),
        likelihood=Gaussian(),
        inducing_variable=Datum.Z,
        num_latent_gps=Datum.num_latent_gps,
        q_diag=diag,
        whiten=whiten,
        q_mu=q_mu,
        q_sqrt=q_sqrt,
    )

    fmean_func, fvar_func = gpflow.conditionals.conditional(
        Datum.X, Datum.Z, model.kernel, model.q_mu, q_sqrt=model.q_sqrt, white=whiten
    )
    mean_value, var_value = fmean_func[0, 0], fvar_func[0, 0]

    assert_allclose(mean_value, Datum.posterior_mean, atol=4)
    assert_allclose(var_value, Datum.posterior_var, atol=4)


@pytest.mark.parametrize("whiten", [True, False])
def test_variational_multivariate_prior_KL_full_q(whiten: bool) -> None:
    cov_q: AnyNDArray = MultiDatum.q_sqrt_full @ MultiDatum.q_sqrt_full.T
    mean_prior = np.zeros((MultiDatum.dim, 1))
    cov_prior = (
        np.eye(MultiDatum.dim)
        if whiten
        else ref_rbf_kernel(MultiDatum.X, MultiDatum.ls, MultiDatum.signal_var)
    )
    reference_kl = multivariate_prior_KL(MultiDatum.q_mean, cov_q, mean_prior, cov_prior)

    q_sqrt = MultiDatum.q_sqrt_full[None, :, :]
    model = gpflow.models.SVGP(
        kernel=SquaredExponential(variance=MultiDatum.signal_var, lengthscales=MultiDatum.ls),
        likelihood=Gaussian(MultiDatum.noise_var),
        inducing_variable=MultiDatum.Z,
        num_latent_gps=MultiDatum.num_latent_gps,
        q_diag=False,
        whiten=whiten,
        q_mu=MultiDatum.q_mean,
        q_sqrt=q_sqrt,
    )

    assert_allclose(model.prior_kl(), reference_kl, atol=4)
