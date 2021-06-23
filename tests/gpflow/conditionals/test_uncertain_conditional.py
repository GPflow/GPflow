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

from collections import namedtuple

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.conditionals import conditional, uncertain_conditional
from gpflow.config import default_float
from gpflow.mean_functions import Constant, Linear, Zero
from gpflow.optimizers import Scipy
from gpflow.quadrature import mvnquad
from gpflow.utilities import training_loop

rng = np.random.RandomState(1)

# ------------------------------------------
# Helpers
# ------------------------------------------


class MomentMatchingSVGP(gpflow.models.SVGP):
    def uncertain_predict_f_moment_matching(self, Xmu, Xcov):
        return uncertain_conditional(
            Xmu,
            Xcov,
            self.inducing_variable,
            self.kernel,
            self.q_mu,
            self.q_sqrt,
            mean_function=self.mean_function,
            white=self.whiten,
            full_output_cov=self.full_output_cov,
        )

    def uncertain_predict_f_monte_carlo(self, Xmu, Xchol, mc_iter=int(1e6)):
        D_in = Xchol.shape[0]
        X_samples = Xmu + np.reshape(
            Xchol[None, :, :] @ rng.randn(mc_iter, D_in)[:, :, None], [mc_iter, D_in]
        )
        F_mu, F_var = self.predict_f(X_samples)
        F_samples = (F_mu + rng.randn(*F_var.shape) * (F_var ** 0.5)).numpy()
        mean = np.mean(F_samples, axis=0)
        covar = np.cov(F_samples.T)
        return mean, covar


def gen_L(n, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])


def gen_q_sqrt(D_out, *shape):
    return tf.convert_to_tensor(
        np.array([np.tril(rng.randn(*shape)) for _ in range(D_out)]), dtype=default_float(),
    )


def mean_function_factory(mean_function_name, D_in, D_out):
    if mean_function_name == "Zero":
        return Zero(output_dim=D_out)
    elif mean_function_name == "Constant":
        return Constant(c=rng.rand(D_out))
    elif mean_function_name == "Linear":
        return Linear(A=rng.rand(D_in, D_out), b=rng.rand(D_out))
    else:
        return None


# ------------------------------------------
# Data classes: storing constants
# ------------------------------------------


class Data:
    N = 7
    N_new = 2
    D_out = 3
    D_in = 1
    X = np.linspace(-5, 5, N)[:, None] + rng.randn(N, 1)
    Y = np.hstack([np.sin(X), np.cos(X), X ** 2])
    Xnew_mu = rng.randn(N_new, 1)
    Xnew_covar = np.zeros((N_new, 1, 1))
    data = (X, Y)


class DataMC1(Data):
    Y = np.hstack([np.sin(Data.X), np.sin(Data.X) * 2, Data.X ** 2])
    data = (Data.X, Y)


class DataMC2(Data):
    N = 7
    N_new = 5
    D_out = 4
    D_in = 2
    X = rng.randn(N, D_in)
    Y = np.hstack([np.sin(X), np.sin(X)])
    Xnew_mu = rng.randn(N_new, D_in)
    L = gen_L(N_new, D_in, D_in)
    Xnew_covar = np.array([l @ l.T for l in L])
    data = (X, Y)


class DataQuad:
    num_data = 10
    num_ind = 10
    D_in = 2
    D_out = 3
    H = 150
    Xmu = tf.convert_to_tensor(rng.randn(num_data, D_in), dtype=default_float())
    L = gen_L(num_data, D_in, D_in)
    Xvar = tf.convert_to_tensor(np.array([l @ l.T for l in L]), dtype=default_float())
    Z = rng.randn(num_ind, D_in)
    q_mu = tf.convert_to_tensor(rng.randn(num_ind, D_out), dtype=default_float())
    q_sqrt = gen_q_sqrt(D_out, num_ind, num_ind)


MEANS = ["Constant", "Linear", "Zero", None]


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("mean", MEANS)
def test_no_uncertainty(white, mean):
    mean_function = mean_function_factory(mean, Data.D_in, Data.D_out)
    kernel = gpflow.kernels.SquaredExponential(variance=rng.rand())
    model = MomentMatchingSVGP(
        kernel,
        gpflow.likelihoods.Gaussian(),
        num_latent_gps=Data.D_out,
        mean_function=mean_function,
        inducing_variable=Data.X.copy(),
        whiten=white,
    )
    model.full_output_cov = False

    training_loop(
        model.training_loss_closure(Data.data),
        optimizer=tf.optimizers.Adam(),
        var_list=model.trainable_variables,
        maxiter=100,
        compile=True,
    )

    mean1, var1 = model.predict_f(Data.Xnew_mu)
    mean2, var2 = model.uncertain_predict_f_moment_matching(
        *map(tf.convert_to_tensor, [Data.Xnew_mu, Data.Xnew_covar])
    )

    assert_allclose(mean1, mean2)
    for n in range(Data.N_new):
        assert_allclose(var1[n, :], var2[n, ...])


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("mean", MEANS)
def test_monte_carlo_1_din(white, mean):
    kernel = gpflow.kernels.SquaredExponential(variance=rng.rand())
    mean_function = mean_function_factory(mean, DataMC1.D_in, DataMC1.D_out)
    model = MomentMatchingSVGP(
        kernel,
        gpflow.likelihoods.Gaussian(),
        num_latent_gps=DataMC1.D_out,
        mean_function=mean_function,
        inducing_variable=DataMC1.X.copy(),
        whiten=white,
    )
    model.full_output_cov = True

    training_loop(
        model.training_loss_closure(DataMC1.data),
        optimizer=tf.optimizers.Adam(),
        var_list=model.trainable_variables,
        maxiter=200,
        compile=True,
    )

    mean1, var1 = model.uncertain_predict_f_moment_matching(
        *map(tf.convert_to_tensor, [DataMC1.Xnew_mu, DataMC1.Xnew_covar])
    )

    for n in range(DataMC1.N_new):
        mean2, var2 = model.uncertain_predict_f_monte_carlo(
            DataMC1.Xnew_mu[n, ...], DataMC1.Xnew_covar[n, ...] ** 0.5
        )
        assert_allclose(mean1[n, ...], mean2, atol=1e-3, rtol=1e-1)
        assert_allclose(var1[n, ...], var2, atol=1e-2, rtol=1e-1)


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("mean", MEANS)
def test_monte_carlo_2_din(white, mean):
    kernel = gpflow.kernels.SquaredExponential(variance=rng.rand())
    mean_function = mean_function_factory(mean, DataMC2.D_in, DataMC2.D_out)
    model = MomentMatchingSVGP(
        kernel,
        gpflow.likelihoods.Gaussian(),
        num_latent_gps=DataMC2.D_out,
        mean_function=mean_function,
        inducing_variable=DataMC2.X.copy(),
        whiten=white,
    )
    model.full_output_cov = True

    training_loop(
        model.training_loss_closure(DataMC2.data),
        optimizer=tf.optimizers.Adam(),
        var_list=model.trainable_variables,
        maxiter=100,
        compile=True,
    )

    mean1, var1 = model.uncertain_predict_f_moment_matching(
        *map(tf.convert_to_tensor, [DataMC2.Xnew_mu, DataMC2.Xnew_covar])
    )

    for n in range(DataMC2.N_new):
        mean2, var2 = model.uncertain_predict_f_monte_carlo(
            DataMC2.Xnew_mu[n, ...], DataMC2.L[n, ...]
        )
        assert_allclose(mean1[n, ...], mean2, atol=1e-2)
        assert_allclose(var1[n, ...], var2, atol=1e-2)


@pytest.mark.parametrize("mean", MEANS)
@pytest.mark.parametrize("white", [True, False])
def test_quadrature(white, mean):
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = gpflow.inducing_variables.InducingPoints(DataQuad.Z)
    mean_function = mean_function_factory(mean, DataQuad.D_in, DataQuad.D_out)

    effective_mean = mean_function or (lambda X: 0.0)

    def conditional_fn(X):
        return conditional(
            X, inducing_variable, kernel, DataQuad.q_mu, q_sqrt=DataQuad.q_sqrt, white=white,
        )

    def mean_fn(X):
        return conditional_fn(X)[0] + effective_mean(X)

    def var_fn(X):
        return conditional_fn(X)[1]

    quad_args = (
        DataQuad.Xmu,
        DataQuad.Xvar,
        DataQuad.H,
        DataQuad.D_in,
        (DataQuad.D_out,),
    )
    mean_quad = mvnquad(mean_fn, *quad_args)
    var_quad = mvnquad(var_fn, *quad_args)

    def mean_sq_fn(X):
        return mean_fn(X) ** 2

    mean_sq_quad = mvnquad(mean_sq_fn, *quad_args)
    var_quad = var_quad + (mean_sq_quad - mean_quad ** 2)

    mean_analytic, var_analytic = uncertain_conditional(
        DataQuad.Xmu,
        DataQuad.Xvar,
        inducing_variable,
        kernel,
        DataQuad.q_mu,
        DataQuad.q_sqrt,
        mean_function=mean_function,
        full_output_cov=False,
        white=white,
    )

    assert_allclose(mean_quad, mean_analytic, rtol=1e-6)
    assert_allclose(var_quad, var_analytic, rtol=1e-6)
