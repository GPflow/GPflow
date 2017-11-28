from collections import namedtuple

import tensorflow as tf

import numpy as np
from numpy.testing import assert_almost_equal

import pytest

import gpflow
from gpflow import settings
from gpflow.conditionals import uncertain_conditional
from gpflow.conditionals import feature_conditional
from gpflow.quadrature import mvnquad
from gpflow.test_util import session_context


class MomentMatchingSVGP(gpflow.models.SVGP):
    @gpflow.params_as_tensors
    @gpflow.autoflow((settings.np_float, [None, None]),
                     (settings.np_float, [None, None, None]))
    def uncertain_predict_f_moment_matching(self, Xmu, Xcov):
        return uncertain_conditional(
            Xmu, Xcov, self.feature, self.kern, self.q_mu, self.q_sqrt,
            whiten=self.whiten, full_cov_output=self.full_cov_output)

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
    q_sqrt = np.array([np.tril(rng.randn(*shape)) for _ in range(D_out)])
    return np.transpose(q_sqrt, [1, 2, 0])


class Data:
    N = 7
    N_new = 2
    D_out = 3
    rng = np.random.RandomState(0)
    X = np.linspace(-5, 5, N)[:, None] + rng.randn(N, 1)
    Y = np.hstack([np.sin(X), np.cos(X), X**2])
    Xnew_mu = rng.randn(N_new, 1)
    Xnew_covar = np.zeros((N_new, 1, 1))


class DataMC1(Data):
    Y = np.hstack([np.sin(Data.X), np.cos(Data.X) * 2, Data.X ** 2])


class DataMC2(Data):
    N = 7
    N_new = 5
    D_out = 4
    D_in = 3

    X = Data.rng.randn(N, D_in)
    Y = Data.rng.randn(N, D_out)
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
    def tensors(cls, white):
        np_float = settings.np_float
        Xmu = tf.placeholder(np_float, [cls.num_data, cls.D_in])
        Xvar = tf.placeholder(np_float, [cls.num_data, cls.D_in, cls.D_in])
        q_mu = tf.placeholder(np_float, [cls.num_ind, cls.D_out])
        q_sqrt = tf.placeholder(np_float, [cls.num_ind, cls.num_ind, cls.D_out])

        kern = gpflow.ekernels.RBF(cls.D_in)
        feat = gpflow.features.InducingPoints(cls.Z)

        feed_dict = {
            Xmu: cls.Xmu,
            Xvar: cls.Xvar,
            q_mu: cls.q_mu,
            q_sqrt: cls.q_sqrt
        }

        def mean_fn(X):
            mean, _ = feature_conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, whiten=white)
            return mean

        def var_fn(X):
            _, var = feature_conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, whiten=white)
            return var

        def mean_sq_fn(X):
            mean, _ = feature_conditional(X, feat, kern, q_mu, q_sqrt=q_sqrt, whiten=white)
            return mean ** 2

        Collection = namedtuple('QuadratureCollection',
                                'Xmu,Xvar,q_mu,q_sqrt,'
                                'kern,feat,feed_dict,mean_fn,'
                                'var_fn,mean_sq_fn')

        return Collection(Xmu=Xmu,
                          Xvar=Xvar,
                          q_mu=q_mu,
                          q_sqrt=q_sqrt,
                          kern=kern,
                          feat=feat,
                          feed_dict=feed_dict,
                          mean_fn=mean_fn,
                          var_fn=var_fn,
                          mean_sq_fn=mean_sq_fn)


@pytest.mark.parametrize('white', [True, False])
@session_context()
def test_no_uncertainty(white):
    k = gpflow.ekernels.RBF(1, variance=Data.rng.rand())
    model = MomentMatchingSVGP(
        Data.X, Data.Y, k, gpflow.likelihoods.Gaussian(),
        Z=Data.X.copy(), whiten=white)
    model.full_cov_output = False
    gpflow.train.AdamOptimizer().minimize(model, maxiter=50)

    mean1, var1 = model.predict_f(Data.Xnew_mu)
    mean2, var2 = model.uncertain_predict_f_moment_matching(Data.Xnew_mu, Data.Xnew_covar)

    assert_almost_equal(mean1, mean2)
    for n in range(Data.N_new):
        assert_almost_equal(var1[n, :], var2[n, ...])


@pytest.mark.parametrize('white', [True, False])
@session_context()
def test_monte_carlo_1_din(white):
    k = gpflow.ekernels.RBF(1, variance=DataMC1.rng.rand())
    model = MomentMatchingSVGP(
        DataMC1.X, DataMC1.Y, k, gpflow.likelihoods.Gaussian(),
        Z=DataMC1.X.copy(), whiten=white)
    model.full_cov_output = True
    gpflow.train.AdamOptimizer().minimize(model, maxiter=50)

    mean1, var1 = model.uncertain_predict_f_moment_matching(DataMC1.Xnew_mu, DataMC1.Xnew_covar)
    for n in range(DataMC1.N_new):
        mean2, var2 = model.uncertain_predict_f_monte_carlo(
            DataMC1.Xnew_mu[n, ...],
            DataMC1.Xnew_covar[n, ...] ** 0.5)
        assert_almost_equal(mean1[n, ...], mean2, decimal=3)
        assert_almost_equal(var1[n, ...], var2, decimal=2)


@pytest.mark.parametrize('white', [True, False])
@session_context()
def test_monte_carlo_2_din(white):
    k = gpflow.ekernels.RBF(DataMC2.D_in, variance=DataMC2.rng.rand())
    model = MomentMatchingSVGP(
        DataMC2.X, DataMC2.Y, k, gpflow.likelihoods.Gaussian(),
        Z=DataMC2.X.copy(), whiten=white)
    model.full_cov_output = True
    gpflow.train.AdamOptimizer().minimize(model)

    mean1, var1 = model.uncertain_predict_f_moment_matching(
        DataMC2.Xnew_mu, DataMC2.Xnew_covar)

    for n in range(DataMC2.N_new):
        mean2, var2 = model.uncertain_predict_f_monte_carlo(
            DataMC2.Xnew_mu[n, ...],
            DataMC2.L[n, ...])
        assert_almost_equal(mean1[n, ...], mean2, decimal=3)
        assert_almost_equal(var1[n, ...], var2, decimal=2)


@pytest.mark.parametrize('white', [True, False])
def test_quadrature_whiten(white):
    with session_context() as session:
        c = DataQuadrature
        d = c.tensors(white)
        quad_args = d.Xmu, d.Xvar, c.H, c.D_in, (c.D_out,)
        mean_quad = mvnquad(d.mean_fn, *quad_args)
        var_quad = mvnquad(d.var_fn, *quad_args)
        mean_sq_quad = mvnquad(d.mean_sq_fn, *quad_args)
        mean_analytic, var_analytic = uncertain_conditional(
            d.Xmu, d.Xvar, d.feat, d.kern,
            d.q_mu, d.q_sqrt,
            full_cov_output=False,
            whiten=white)

        mean_quad, var_quad, mean_sq_quad = session.run(
            [mean_quad, var_quad, mean_sq_quad], feed_dict=d.feed_dict)
        var_quad = var_quad + (mean_sq_quad - mean_quad**2)
        mean_analytic, var_analytic = session.run(
            [mean_analytic, var_analytic], feed_dict=d.feed_dict)

        assert_almost_equal(mean_quad, mean_analytic, decimal=6)
        assert_almost_equal(var_quad, var_analytic, decimal=6)


if __name__ == "__main__":
    tf.test.main()
