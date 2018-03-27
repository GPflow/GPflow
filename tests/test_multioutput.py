import pytest
import numpy as np
import gpflow
import gpflow.multioutput.kernels as mk
import gpflow.multioutput.features as mf

from gpflow.models import SVGP
from gpflow.kernels import RBF, Linear
from gpflow.features import InducingPoints
from gpflow.likelihoods import Gaussian

class Data:
    X = np.random.rand(100)[:, None] * 10 - 5
    G = np.hstack((0.5 * np.sin(3 * X) + X, 3.0 * np.cos(X) - X))
    Ptrue = np.array([[0.5, -0.3, 1.5], [-0.4, 0.43, 0.0]])
    Y = np.matmul(G, Ptrue)
    Y += np.random.randn(*Y.shape) * [0.2, 0.2, 0.2]
    Xs = np.linspace(-6, 6, 20)[:, None]

    D = 1
    M = 4
    L = 2
    P = 3
    MAXITER = int(15e1)

def test_1():
    q_mu = np.random.randn(Data.M * Data.P, 1)  # MP x 1
    q_sqrt = np.tril(np.random.randn(Data.M * Data.P, Data.M * Data.P))[None, ...]  # 1 x MP x MP
    print(q_sqrt[0, Data.M:2*Data.M, Data.M:2*Data.M])
    kernel = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
    feature = InducingPoints(Data.X[:Data.M,...].copy())
    m1 = SVGP(Data.X, Data.Y, kernel, Gaussian(), feature, q_mu=q_mu.copy(), q_sqrt=q_sqrt.copy())

    q_sqrt = np.transpose(q_sqrt[0, ...].reshape([Data.P, Data.M, Data.P, Data.M]), [1, 3, 0, 2])  # M x M x P x P
    q_sqrt = np.array([q_sqrt[..., p, p] for p in range(Data.P)])  # P x M x M
    print(q_sqrt[1, ...])
    q_mu = q_mu.reshape(Data.M, Data.P)  # M x P
    kernel2 = RBF(Data.D, variance=0.5, lengthscales=1.2)
    m2 = SVGP(Data.X, Data.Y, kernel2, Gaussian(), Z=Data.X[:Data.M, ...].copy(), q_mu=q_mu.copy(), q_sqrt=q_sqrt.copy())

    y1_s_mean, y1_s_var = m1.predict_f(Data.Xs)
    y2_s_mean, y2_s_var = m2.predict_f(Data.Xs)
    # print(y1_s_mean)
    # print(y2_s_mean)
    # print(y1_s_var)
    # print(y2_s_var)
    
    np.testing.assert_almost_equal(m1.predict_f(Data.Xs), m2.predict_f(Data.Xs), decimal=3)


# def test_2():
#     kernel = mk.SharedIndependentMok(RBF(Data.D, variance=0.5, lengthscales=1.2), Data.P)
#     feature = mf.SharedIndependentMof(InducingPoints(Data.X[:Data.M,...].copy()))
#     m1 = SVGP(Data.X, Data.Y, kernel, Gaussian(), feat=feature)

#     m2 = SVGP(Data.X, Data.Y, RBF(Data.D, variance=0.5, lengthscales=1.2), Gaussian(), Z=Data.X[:Data.M, ...].copy())

#     np.testing.assert_allclose(m1.predict_f(Data.Xs), m2.predict_f(Data.Xs))







# if __name__ == "__main__":
#     tf.test.main()

