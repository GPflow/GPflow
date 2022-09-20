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

from typing import Collection, Optional

import numpy as np
import pytest
import tensorflow as tf
from numpy.testing import assert_allclose

import gpflow
from gpflow.base import AnyNDArray, MeanAndVariance
from gpflow.conditionals import conditional, uncertain_conditional
from gpflow.config import default_float
from gpflow.experimental.check_shapes import ShapeChecker, check_shapes
from gpflow.mean_functions import Constant, Linear, MeanFunction, Zero
from gpflow.quadrature import mvnquad
from gpflow.utilities import training_loop

rng = np.random.RandomState(1)

# ------------------------------------------
# Helpers
# ------------------------------------------


class MomentMatchingSVGP(gpflow.models.SVGP):
    @check_shapes(
        "Xmu: [batch..., N, Din]",
        "Xcov: [batch..., N, n, n]",
        "self.inducing_variable: [M, Din, broadcast t]",
        "self.q_mu: [M, Dout]",
        "self.q_sqrt: [t, M, M]",
        "return[0]: [batch..., N, Dout]",
        "return[1]: [batch..., N, t, t] if self.full_output_cov",
        "return[1]: [batch..., N, Dout] if not self.full_output_cov",
    )
    def uncertain_predict_f_moment_matching(
        self, Xmu: tf.Tensor, Xcov: tf.Tensor
    ) -> MeanAndVariance:
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

    @check_shapes(
        "Xmu: [Din]",
        "Xchol: [n, n]",
        "return[0]: [Dout]",
        "return[1]: [t, t] if self.full_output_cov",
        "return[1]: [Dout] if not self.full_output_cov",
    )
    def uncertain_predict_f_monte_carlo(
        self, Xmu: tf.Tensor, Xchol: tf.Tensor, mc_iter: int = int(1e6)
    ) -> MeanAndVariance:
        D_in = Xchol.shape[0]
        X_samples = Xmu + np.reshape(
            Xchol[None, :, :] @ rng.randn(mc_iter, D_in)[:, :, None], [mc_iter, D_in]
        )
        F_mu, F_var = self.predict_f(X_samples)
        F_samples = (F_mu + rng.randn(*F_var.shape) * (F_var ** 0.5)).numpy()
        mean = np.mean(F_samples, axis=0)
        covar = np.cov(F_samples.T)
        return mean, covar


@check_shapes(
    "return: [n, shape...]",
)
def gen_L(n: int, *shape: int) -> AnyNDArray:
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])


def gen_q_sqrt(D_out: int, *shape: int) -> tf.Tensor:
    return tf.convert_to_tensor(
        np.array([np.tril(rng.randn(*shape)) for _ in range(D_out)]),
        dtype=default_float(),
    )


def mean_function_factory(
    mean_function_name: Optional[str], D_in: int, D_out: int
) -> Optional[MeanFunction]:
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
    cs = ShapeChecker().check_shape

    N = 7
    N_new = 2
    D_out = 3
    D_in = 1
    X = cs(np.linspace(-5, 5, N)[:, None] + rng.randn(N, 1), "[N, D_in]")
    Y: AnyNDArray = cs(np.hstack([np.sin(X), np.cos(X), X ** 2]), "[N, D_out]")
    Xnew_mu = cs(rng.randn(N_new, 1), "[N_new, D_in]")
    Xnew_covar = cs(np.zeros((N_new, 1, 1)), "[N_new, D_in, D_in]")
    data = (X, Y)


class DataMC1(Data):
    Y: AnyNDArray = Data.cs(  # type: ignore[misc]
        np.hstack([np.sin(Data.X), np.sin(Data.X) * 2, Data.X ** 2]), "[N, D_out]"
    )
    data = (Data.X, Y)


class DataMC2(Data):
    cs = ShapeChecker().check_shape

    N = 7
    N_new = 5
    D_out = 4
    D_in = 2
    X = cs(rng.randn(N, D_in), "[N, D_in]")
    Y: AnyNDArray = cs(np.hstack([np.sin(X), np.sin(X)]), "[N, D_out]")
    Xnew_mu = cs(rng.randn(N_new, D_in), "[N_new, D_in]")
    L = cs(gen_L(N_new, D_in, D_in), "[N_new, D_in, D_in]")
    Xnew_covar: AnyNDArray = cs(np.array([l @ l.T for l in L]), "[N_new, D_in, D_in]")
    data = (X, Y)


class DataQuad:
    cs = ShapeChecker().check_shape

    num_data = 10
    num_ind = 10
    D_in = 2
    D_out = 3
    H = 150
    Xmu = cs(
        tf.convert_to_tensor(rng.randn(num_data, D_in), dtype=default_float()), "[num_data, D_in]"
    )
    L = cs(gen_L(num_data, D_in, D_in), "[num_data, D_in, D_in]")
    Xvar = cs(
        tf.convert_to_tensor(np.array([l @ l.T for l in L]), dtype=default_float()),
        "[num_data, D_in, D_in]",
    )
    Z = cs(rng.randn(num_ind, D_in), "[num_ind, D_in]")
    q_mu = cs(
        tf.convert_to_tensor(rng.randn(num_ind, D_out), dtype=default_float()), "[num_ind, D_out]"
    )
    q_sqrt = cs(gen_q_sqrt(D_out, num_ind, num_ind), "[D_out, num_ind, num_ind]")


MEANS: Collection[Optional[str]] = ["Constant", "Linear", "Zero", None]


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("mean", MEANS)
def test_no_uncertainty(white: bool, mean: Optional[str]) -> None:
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
def test_monte_carlo_1_din(white: bool, mean: Optional[str]) -> None:
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
def test_monte_carlo_2_din(white: bool, mean: Optional[str]) -> None:
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


@pytest.mark.parametrize("white", [True, False])
@pytest.mark.parametrize("mean", MEANS)
def test_quadrature(white: bool, mean: Optional[str]) -> None:
    kernel = gpflow.kernels.SquaredExponential()
    inducing_variable = gpflow.inducing_variables.InducingPoints(DataQuad.Z)
    mean_function = mean_function_factory(mean, DataQuad.D_in, DataQuad.D_out)

    effective_mean = mean_function or (lambda X: 0.0)

    def conditional_fn(X: tf.Tensor) -> tf.Tensor:
        return conditional(
            X,
            inducing_variable,
            kernel,
            DataQuad.q_mu,
            q_sqrt=DataQuad.q_sqrt,
            white=white,
        )

    def mean_fn(X: tf.Tensor) -> tf.Tensor:
        return conditional_fn(X)[0] + effective_mean(X)

    def var_fn(X: tf.Tensor) -> tf.Tensor:
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

    def mean_sq_fn(X: tf.Tensor) -> tf.Tensor:
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
