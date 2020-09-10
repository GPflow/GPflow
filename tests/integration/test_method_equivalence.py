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

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.config import default_jitter
from gpflow.mean_functions import Constant
from gpflow.models import maximum_log_likelihood_objective

rng = np.random.RandomState(0)


class Datum:
    X = rng.rand(20, 1) * 10
    Y = np.sin(X) + 0.9 * np.cos(X * 1.6) + rng.randn(*X.shape) * 0.8
    Y = np.tile(Y, 2)  # two identical columns
    Xtest = rng.rand(10, 1) * 10
    data = (X, Y)


class DatumVGP:
    N, Ns, DX, DY = 100, 10, 2, 2
    np.random.seed(1)
    X = np.random.randn(N, DX)
    Xs = np.random.randn(Ns, DX)
    Y = np.random.randn(N, DY)
    q_mu = np.random.randn(N, DY)
    q_sqrt = np.random.randn(DY, N, N)
    q_alpha = np.random.randn(N, DX)
    q_lambda = np.random.randn(N, DX) ** 2
    data = (X, Y)


def _create_full_gp_model():
    """
    GP Regression
    """
    full_gp_model = gpflow.models.GPR(
        (Datum.X, Datum.Y),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=gpflow.mean_functions.Constant(),
    )

    opt = gpflow.optimizers.Scipy()
    opt.minimize(
        full_gp_model.training_loss,
        variables=full_gp_model.trainable_variables,
        options=dict(maxiter=300),
    )
    return full_gp_model


def _create_approximate_models():
    """
    1) Variational GP (with the likelihood set to Gaussian)
    2) Sparse variational GP (likelihood is Gaussian, inducing points
       at the data)
    3) Sparse variational GP (as above, but with the whitening rotation
       of the inducing variables)
    4) Sparse variational GP Regression (as above, but there the inducing
       variables are 'collapsed' out, as in Titsias 2009)
    5) FITC Sparse GP Regression
    """
    model_1 = gpflow.models.VGP(
        (Datum.X, Datum.Y),
        gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Gaussian(),
        mean_function=gpflow.mean_functions.Constant(),
    )
    model_2 = gpflow.models.SVGP(
        gpflow.kernels.SquaredExponential(),
        gpflow.likelihoods.Gaussian(),
        inducing_variable=Datum.X.copy(),
        q_diag=False,
        mean_function=gpflow.mean_functions.Constant(),
        num_latent_gps=Datum.Y.shape[1],
    )
    gpflow.set_trainable(model_2.inducing_variable, False)
    model_3 = gpflow.models.SVGP(
        kernel=gpflow.kernels.SquaredExponential(),
        likelihood=gpflow.likelihoods.Gaussian(),
        inducing_variable=Datum.X.copy(),
        q_diag=False,
        whiten=True,
        mean_function=gpflow.mean_functions.Constant(),
        num_latent_gps=Datum.Y.shape[1],
    )
    gpflow.set_trainable(model_3.inducing_variable, False)
    model_4 = gpflow.models.GPRFITC(
        (Datum.X, Datum.Y),
        kernel=gpflow.kernels.SquaredExponential(),
        inducing_variable=Datum.X.copy(),
        mean_function=Constant(),
    )
    gpflow.set_trainable(model_4.inducing_variable, False)
    model_5 = gpflow.models.SGPR(
        (Datum.X, Datum.Y),
        gpflow.kernels.SquaredExponential(),
        inducing_variable=Datum.X.copy(),
        mean_function=Constant(),
    )
    gpflow.set_trainable(model_5.inducing_variable, False)

    # Train models

    opt = gpflow.optimizers.Scipy()

    opt.minimize(
        model_1.training_loss, variables=model_1.trainable_variables, options=dict(maxiter=300),
    )
    opt.minimize(
        model_2.training_loss_closure(Datum.data),
        variables=model_2.trainable_variables,
        options=dict(maxiter=300),
    )
    opt.minimize(
        model_3.training_loss_closure(Datum.data),
        variables=model_3.trainable_variables,
        options=dict(maxiter=300),
    )
    opt.minimize(
        model_4.training_loss, variables=model_4.trainable_variables, options=dict(maxiter=300),
    )
    opt.minimize(
        model_5.training_loss, variables=model_5.trainable_variables, options=dict(maxiter=300),
    )

    return model_1, model_2, model_3, model_4, model_5


def _create_vgp_model(kernel, likelihood, q_mu=None, q_sqrt=None):
    model_vgp = gpflow.models.VGP((DatumVGP.X, DatumVGP.Y), kernel, likelihood)
    if q_mu is not None and q_sqrt is not None:
        model_vgp.q_mu.assign(q_mu)
        model_vgp.q_sqrt.assign(q_sqrt)
    return model_vgp


def _create_vgpao_model(kernel, likelihood, q_alpha, q_lambda):
    model_vgpoa = gpflow.models.VGPOpperArchambeau(
        (DatumVGP.X, DatumVGP.Y), kernel, likelihood, num_latent_gps=DatumVGP.DY
    )
    model_vgpoa.q_alpha.assign(q_alpha)
    model_vgpoa.q_lambda.assign(q_lambda)

    return model_vgpoa


def _create_svgp_model(kernel, likelihood, q_mu, q_sqrt, whiten):
    model_svgp = gpflow.models.SVGP(
        kernel,
        likelihood,
        DatumVGP.X.copy(),
        whiten=whiten,
        q_diag=False,
        num_latent_gps=DatumVGP.DY,
    )
    model_svgp.q_mu.assign(q_mu)
    model_svgp.q_sqrt.assign(q_sqrt)
    return model_svgp


@pytest.mark.parametrize("approximate_model", _create_approximate_models())
def test_equivalence(approximate_model):
    """
    With a Gaussian likelihood, and inducing points (where appropriate)
    positioned at the data, many of the gpflow methods are equivalent (perhaps
    subject to some optimization).
    """
    gpr_model = _create_full_gp_model()
    gpr_likelihood = gpr_model.log_marginal_likelihood()
    approximate_likelihood = maximum_log_likelihood_objective(approximate_model, Datum.data)
    assert_allclose(approximate_likelihood, gpr_likelihood, rtol=1e-6)

    gpr_kernel_ls = gpr_model.kernel.lengthscales.numpy()
    gpr_kernel_var = gpr_model.kernel.variance.numpy()

    approximate_kernel_ls = approximate_model.kernel.lengthscales.numpy()
    approximate_kernel_var = approximate_model.kernel.variance.numpy()

    assert_allclose(gpr_kernel_ls, approximate_kernel_ls, 1e-4)
    assert_allclose(gpr_kernel_var, approximate_kernel_var, 1e-3)

    gpr_mu, gpr_var = gpr_model.predict_y(Datum.Xtest)
    approximate_mu, approximate_var = approximate_model.predict_y(Datum.Xtest)

    assert_allclose(gpr_mu, approximate_mu, 1e-3)
    assert_allclose(gpr_var, approximate_var, 1e-4)


def test_equivalence_vgp_and_svgp():
    kernel = gpflow.kernels.Matern52()
    likelihood = gpflow.likelihoods.StudentT()

    svgp_model = _create_svgp_model(kernel, likelihood, DatumVGP.q_mu, DatumVGP.q_sqrt, whiten=True)
    vgp_model = _create_vgp_model(kernel, likelihood, DatumVGP.q_mu, DatumVGP.q_sqrt)

    likelihood_svgp = svgp_model.elbo(DatumVGP.data)
    likelihood_vgp = vgp_model.elbo()
    assert_allclose(likelihood_svgp, likelihood_vgp, rtol=1e-2)

    svgp_mu, svgp_var = svgp_model.predict_f(DatumVGP.Xs)
    vgp_mu, vgp_var = vgp_model.predict_f(DatumVGP.Xs)

    assert_allclose(svgp_mu, vgp_mu)
    assert_allclose(svgp_var, vgp_var)


def test_equivalence_vgp_and_opper_archambeau():
    kernel = gpflow.kernels.Matern52()
    likelihood = gpflow.likelihoods.StudentT()

    vgp_oa_model = _create_vgpao_model(kernel, likelihood, DatumVGP.q_alpha, DatumVGP.q_lambda)

    K = kernel(DatumVGP.X) + np.eye(DatumVGP.N) * default_jitter()
    L = np.linalg.cholesky(K)
    L_inv = np.linalg.inv(L)
    K_inv = np.linalg.inv(K)

    mean = K @ DatumVGP.q_alpha

    prec_dnn = K_inv[None, :, :] + np.array([np.diag(l ** 2) for l in DatumVGP.q_lambda.T])
    var_dnn = np.linalg.inv(prec_dnn)

    svgp_model_unwhitened = _create_svgp_model(
        kernel, likelihood, mean, np.linalg.cholesky(var_dnn), whiten=False
    )

    mean_white_nd = L_inv.dot(mean)
    var_white_dnn = np.einsum("nN,dNM,mM->dnm", L_inv, var_dnn, L_inv)
    q_sqrt_nnd = np.linalg.cholesky(var_white_dnn)

    vgp_model = _create_vgp_model(kernel, likelihood, mean_white_nd, q_sqrt_nnd)

    likelihood_vgp = vgp_model.elbo()
    likelihood_vgp_oa = vgp_oa_model.elbo()
    likelihood_svgp_unwhitened = svgp_model_unwhitened.elbo(DatumVGP.data)

    assert_allclose(likelihood_vgp, likelihood_vgp_oa, rtol=1e-2)
    assert_allclose(likelihood_vgp, likelihood_svgp_unwhitened, rtol=1e-2)

    vgp_oa_mu, vgp_oa_var = vgp_oa_model.predict_f(DatumVGP.Xs)
    svgp_unwhitened_mu, svgp_unwhitened_var = svgp_model_unwhitened.predict_f(DatumVGP.Xs)
    vgp_mu, vgp_var = vgp_model.predict_f(DatumVGP.Xs)

    assert_allclose(vgp_oa_mu, vgp_mu)
    assert_allclose(vgp_oa_var, vgp_var, rtol=1e-4)  # jitter?
    assert_allclose(svgp_unwhitened_mu, vgp_mu)
    assert_allclose(svgp_unwhitened_var, vgp_var, rtol=1e-4)


class DatumUpper:
    rng = np.random.default_rng(123)
    X = rng.random((100, 1))
    Y = np.sin(1.5 * 2 * np.pi * X) + rng.standard_normal(X.shape) * 0.1 + 5.3
    assert Y.mean() > 5.0, "offset ensures a regression test against the bug fixed by PR #1560"
    data = (X, Y)


def test_upper_bound_few_inducing_points():
    """
    Test for upper bound for regression marginal likelihood
    """
    model_vfe = gpflow.models.SGPR(
        (DatumUpper.X, DatumUpper.Y),
        gpflow.kernels.SquaredExponential(),
        inducing_variable=DatumUpper.X[:10, :].copy(),
        mean_function=Constant(),
    )
    opt = gpflow.optimizers.Scipy()

    opt.minimize(
        model_vfe.training_loss, variables=model_vfe.trainable_variables, options=dict(maxiter=500),
    )

    full_gp = gpflow.models.GPR(
        (DatumUpper.X, DatumUpper.Y),
        kernel=gpflow.kernels.SquaredExponential(),
        mean_function=Constant(),
    )
    full_gp.kernel.lengthscales.assign(model_vfe.kernel.lengthscales)
    full_gp.kernel.variance.assign(model_vfe.kernel.variance)
    full_gp.likelihood.variance.assign(model_vfe.likelihood.variance)
    full_gp.mean_function.c.assign(model_vfe.mean_function.c)

    lml_upper = model_vfe.upper_bound()
    lml_vfe = model_vfe.elbo()
    lml_full_gp = full_gp.log_marginal_likelihood()

    assert lml_vfe < lml_full_gp
    assert lml_full_gp < lml_upper
