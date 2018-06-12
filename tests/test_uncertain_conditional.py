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

import tensorflow as tf

import numpy as np
from numpy.testing import assert_almost_equal

import pytest

import gpflow
from gpflow import settings
from gpflow.conditionals import conditional
from gpflow.conditionals import uncertain_conditional
from gpflow.quadrature import mvnquad
from gpflow.test_util import session_context


class MomentMatchingSVGP(gpflow.models.SVGP):
    @gpflow.params_as_tensors
    def uncertain_predict_f_moment_matching(self, Xmu, Xcov):
        return uncertain_conditional(
                Xmu, Xcov, self.feature, self.kern, self.q_mu, self.q_sqrt,
                mean_function=self.mean_function, white=self.whiten,
                full_output_cov=self.full_output_cov)

    def uncertain_predict_f_monte_carlo(self, Xmu, Xchol, mc_iter=int(1e6)):
        rng = np.random.RandomState(0)
        D_in = Xchol.shape[0]
        X_samples = Xmu + np.reshape(
            Xchol[None, :, :] @ rng.randn(mc_iter, D_in)[:, :, None], [mc_iter, D_in])
        F_mu, F_var = self.predict_f(X_samples)
        F_samples = F_mu + rng.randn(*F_var.shape) * (F_var ** 0.5)
        mean = np.mean(F_samples, axis=0)
        covar = np.cov(F_samples.T)
        return mean, covar


def gen_L(rng, n, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(n)])


def gen_q_sqrt(rng, D_out, *shape):
    return np.array([np.tril(rng.randn(*shape)) for _ in range(D_out)])


def mean_function_factory(rng, mean_function_name, D_in, D_out):
    if mean_function_name == "Zero":
        return gpflow.mean_functions.Zero(output_dim=D_out)
    elif mean_function_name == "Constant":
        return gpflow.mean_functions.Constant(c=rng.rand(D_out))
    elif mean_function_name == "Linear":
        return gpflow.mean_functions.Linear(
                A=rng.rand(D_in, D_out), b=rng.rand(D_out))
    else:
        return None


class Data:
    N = 7
    N_new = 2
    D_out = 3
    D_in = 1
    rng = np.random.RandomState(1)
    X = np.linspace(-5, 5, N)[:, None] + rng.randn(N, 1)
    Y = np.hstack([np.sin(X), np.cos(X), X**2])
    Xnew_mu = rng.randn(N_new, 1)
    Xnew_covar = np.zeros((N_new, 1, 1))


class DataMC1(Data):
    Y = np.hstack([np.sin(Data.X), np.sin(Data.X) * 2, Data.X ** 2])


class DataMC2(Data):
    N = 7
    N_new = 5
    D_out = 4
    D_in = 2

    X = Data.rng.randn(N, D_in)
    Y = np.hstack([np.sin(X), np.sin(X)])
    Xnew_mu = Data.rng.randn(N_new, D_in)
    L = gen_L(Data.rng, N_new, D_in, D_in)
    Xnew_covar = np.array([l @ l.T for l in L])


class DataQuadrature:
    num_data = 10
    num_ind = 10
    D_in = 2
    D_out = 3
    H = 150

    rng = np.random.RandomState(1)

    Xmu = rng.randn(num_data, D_in)
    L = gen_L(rng, num_data, D_in, D_in)
    Xvar = np.array([l @ l.T for l in L])
    Z = rng.randn(num_ind, D_in)
    q_mu = rng.randn(num_ind, D_out)
    q_sqrt = gen_q_sqrt(rng, D_out, num_ind, num_ind)

    @classmethod
    def tensors(cls, white, mean_name):
        float_type = settings.float_type
        Xmu = tf.placeholder(float_type, [cls.num_data, cls.D_in])
        Xvar = tf.placeholder(float_type, [cls.num_data, cls.D_in, cls.D_in])
        q_mu = tf.placeholder(float_type, [cls.num_ind, cls.D_out])
        q_sqrt = tf.placeholder(float_type, [cls.D_out, cls.num_ind, cls.num_ind])

        kern = gpflow.kernels.RBF(cls.D_in)
        feat = gpflow.features.InducingPoints(cls.Z)
        mean_function = mean_function_factory(cls.rng, mean_name, cls.D_in, cls.D_out)
        effective_mean = mean_function or (lambda X: 0.0)

        feed_dict = {
            Xmu: cls.Xmu,
            Xvar: cls.Xvar,
            q_mu: cls.q_mu,
            q_sqrt: cls.q_sqrt
        }

        def mean_fn(X):
            mean, _ = conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, white=white)
            return mean + effective_mean(X)

        def var_fn(X):
            _, var = conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, white=white)
            return var

        def mean_sq_fn(X):
            mean, _ = conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, white=white)
            return (mean + effective_mean(X)) ** 2

        Collection = namedtuple('QuadratureCollection',
                                'Xmu,Xvar,q_mu,q_sqrt,'
                                'kern,feat,mean_function,'
                                'feed_dict,mean_fn,'
                                'var_fn,mean_sq_fn')

        return Collection(Xmu=Xmu,
                          Xvar=Xvar,
                          q_mu=q_mu,
                          q_sqrt=q_sqrt,
                          kern=kern,
                          feat=feat,
                          mean_function=mean_function,
                          feed_dict=feed_dict,
                          mean_fn=mean_fn,
                          var_fn=var_fn,
                          mean_sq_fn=mean_sq_fn)


MEANS = ["Constant", "Linear", "Zero", None]

@pytest.mark.parametrize('white', [True, False])
@pytest.mark.parametrize('mean', MEANS)
def test_no_uncertainty(white, mean):
    with session_context() as sess:
        m = mean_function_factory(Data.rng, mean, Data.D_in, Data.D_out)
        k = gpflow.kernels.RBF(1, variance=Data.rng.rand())
        model = MomentMatchingSVGP(
            Data.X, Data.Y, k, gpflow.likelihoods.Gaussian(),
            mean_function=m, Z=Data.X.copy(), whiten=white)
        model.full_output_cov = False
        gpflow.train.AdamOptimizer().minimize(model, maxiter=50)

        mean1, var1 = model.predict_f(Data.Xnew_mu)
        pred_mm = model.uncertain_predict_f_moment_matching(
                            tf.constant(Data.Xnew_mu), tf.constant(Data.Xnew_covar))
        mean2, var2 = sess.run(pred_mm)

        assert_almost_equal(mean1, mean2)
        for n in range(Data.N_new):
            assert_almost_equal(var1[n, :], var2[n, ...])


@pytest.mark.parametrize('white', [True, False])
@pytest.mark.parametrize('mean', MEANS)
def test_monte_carlo_1_din(white, mean):
    with session_context() as sess:
        k = gpflow.kernels.RBF(1, variance=DataMC1.rng.rand())
        m = mean_function_factory(DataMC1.rng, mean, DataMC1.D_in, DataMC1.D_out)
        model = MomentMatchingSVGP(
            DataMC1.X, DataMC1.Y, k, gpflow.likelihoods.Gaussian(),
            Z=DataMC1.X.copy(), whiten=white)
        model.full_output_cov = True
        gpflow.train.AdamOptimizer().minimize(model, maxiter=50)

        pred_mm = model.uncertain_predict_f_moment_matching(
                            tf.constant(DataMC1.Xnew_mu), tf.constant(DataMC1.Xnew_covar))
        mean1, var1 = sess.run(pred_mm)

        for n in range(DataMC1.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(
                DataMC1.Xnew_mu[n, ...],
                DataMC1.Xnew_covar[n, ...] ** 0.5)
            assert_almost_equal(mean1[n, ...], mean2, decimal=3)
            assert_almost_equal(var1[n, ...], var2, decimal=2)


@pytest.mark.parametrize('white', [True, False])
@pytest.mark.parametrize('mean', MEANS)
def test_monte_carlo_2_din(white, mean):
    with session_context() as sess:
        k = gpflow.kernels.RBF(DataMC2.D_in, variance=DataMC2.rng.rand())
        m = mean_function_factory(DataMC2.rng, mean, DataMC2.D_in, DataMC2.D_out)
        model = MomentMatchingSVGP(
            DataMC2.X, DataMC2.Y, k, gpflow.likelihoods.Gaussian(),
            mean_function=m, Z=DataMC2.X.copy(), whiten=white)
        model.full_output_cov = True
        gpflow.train.AdamOptimizer().minimize(model)

        pred_mm = model.uncertain_predict_f_moment_matching(
                            tf.constant(DataMC2.Xnew_mu), tf.constant(DataMC2.Xnew_covar))
        mean1, var1 = sess.run(pred_mm)

        for n in range(DataMC2.N_new):
            mean2, var2 = model.uncertain_predict_f_monte_carlo(
                DataMC2.Xnew_mu[n, ...],
                DataMC2.L[n, ...])
            assert_almost_equal(mean1[n, ...], mean2, decimal=2)
            assert_almost_equal(var1[n, ...], var2, decimal=2)


@pytest.mark.parametrize('mean', MEANS)
@pytest.mark.parametrize('white', [True, False])
def test_quadrature(white, mean):
    with session_context() as session:
        c = DataQuadrature
        d = c.tensors(white, mean)
        quad_args = d.Xmu, d.Xvar, c.H, c.D_in, (c.D_out,)
        mean_quad = mvnquad(d.mean_fn, *quad_args)
        var_quad = mvnquad(d.var_fn, *quad_args)
        mean_sq_quad = mvnquad(d.mean_sq_fn, *quad_args)
        mean_analytic, var_analytic = uncertain_conditional(
            d.Xmu, d.Xvar, d.feat, d.kern,
            d.q_mu, d.q_sqrt,
            mean_function=d.mean_function,
            full_output_cov=False,
            white=white)

        mean_quad, var_quad, mean_sq_quad = session.run(
            [mean_quad, var_quad, mean_sq_quad], feed_dict=d.feed_dict)
        var_quad = var_quad + (mean_sq_quad - mean_quad**2)
        mean_analytic, var_analytic = session.run(
            [mean_analytic, var_analytic], feed_dict=d.feed_dict)

        assert_almost_equal(mean_quad, mean_analytic, decimal=6)
        assert_almost_equal(var_quad, var_analytic, decimal=6)


if __name__ == "__main__":
    tf.test.main()
