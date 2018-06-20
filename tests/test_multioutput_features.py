import gpflow
import gpflow.multioutput.features as mf
import gpflow.multioutput.kernels as mk
import numpy as np
import pytest
import tensorflow as tf
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.models import SVGP
from gpflow.test_util import session_tf


float_type = gpflow.settings.float_type
np.random.seed(1)


class Datum:
    D = 1
    L = 2
    P = 3
    M = 10
    N = 100
    W = np.random.randn(P, L)
    X = np.random.randn(N)[:, None]
    Xnew = np.random.randn(N)[:, None]


def make_kernel():
    return gpflow.kernels.RBF(Datum.D)


def make_kernels(num):
    return [make_kernel() for _ in range(num)]


def make_ip():
    x = np.random.permutation(Datum.X)
    return gpflow.features.InducingPoints(x[:Datum.M, ...])


def make_ips(num):
    return [make_ip() for _ in range(num)]


class Mofs:
    def shared_independent(self):
        return mf.SharedIndependentMof(make_ip())

    def separate_independent(self, num=Datum.P):
        return mf.SeparateIndependentMof(make_ips(num))

    def features(self):
        return [self.shared_independent, self.separate_independent]

    def mixed_shared(self):
        return mf.MixedKernelSharedMof(make_ip())


class Moks:
    def shared_independent(self):
        return mk.SharedIndependentMok(make_kernel(), Datum.P)

    def separate_independent(self, num=Datum.L):
        return mk.SeparateIndependentMok(make_kernels(num))

    def separate_mixed(self, num=Datum.L):
        return mk.SeparateMixedMok(make_kernels(num), Datum.W)

    def kernels(self):
        return [self.shared_independent, self.separate_independent, self.separate_mixed]


@pytest.mark.parametrize('feature', Mofs().features())
@pytest.mark.parametrize('kernel', Moks().kernels())
def test_kuu(session_tf, feature, kernel):
    Kuu = mf.Kuu(feature(), kernel(), jitter=1e-9)
    session_tf.run(tf.cholesky(Kuu))


@pytest.mark.parametrize('feature', Mofs().features())
@pytest.mark.parametrize('kernel', Moks().kernels())
def test_kuf(session_tf, feature, kernel):
    Kuf = mf.Kuf(feature(), kernel(), Datum.Xnew)
    session_tf.run(Kuf)


@pytest.mark.parametrize('fun', [mf.Kuu, mf.Kuf])
def test_mixed_shared(session_tf, fun):
    f = Mofs().mixed_shared()
    k = Moks().separate_mixed()
    if fun is mf.Kuu:
        t = tf.cholesky(fun(f, k, jitter=1e-9))
    else:
        t = fun(f, k, Datum.Xnew)
        print(t.shape)
    session_tf.run(t)
