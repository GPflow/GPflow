import numpy as np
import tensorflow as tf
import gpflow
from gpflow.features import InducingPoints
from gpflow.kernels import RBF
from gpflow.probability_distributions import MarkovGaussian
from gpflow.mean_functions import Identity
from gpflow.expectations_quadrature import quadrature_expectation
from gpflow.expectations import expectation

if __name__ == '__main__':
    rng = np.random.RandomState(1)
    Z = rng.randn(10, 2) * .1
    feat = InducingPoints(Z)
    mean = Identity(2)

    Xmu = rng.randn(20, 2) * .1
    
    L = np.array([np.tril(rng.randn(2, 2)) for _ in range(20)])
    L2 = np.array([np.tril(rng.randn(2, 2)) for _ in range(19)])
    
    LLleft = np.concatenate((L[:-1], L[1:]), 1)
    LL = np.concatenate((LLleft, np.zeros((19, 4, 2))), 2)
    LL += 1e-3*np.eye(4, 4)
    Xcov = LL @ np.transpose(LL, (0, 2, 1))
    # LLup = np.concatenate((L[:-1], np.zeros((19, 2, 2))), 1)
    # LLdown = np.concatenate((L[1:], np.zeros(19, 2, 2))), 1)
    # LL = np.concatenate((LLup, LLdown), 1)
    # Xcov = np.array([l @ l.T for l in L])
    # Xcov2 = np.array([l @ l.T for l in L2])
    # Xcc = np.stack([Xcov, Xcov2])

    # LL = np.array([np.tril(rng.randn(4, 4)) for _ in range(19)])
    # Xcov = LL @ np.transpose(LL, (0, 2, 1))

    Xc = np.concatenate((Xcov[:, :2, :2], Xcov[-1:, 2:, 2:]), 0)
    Xcross = np.concatenate((Xcov[:, 0:2, 2:], np.zeros((1, 2, 2))), 0)
    Xcc = np.stack([Xc, Xcross])

    markovgauss = MarkovGaussian(Xmu, Xcc)

    with gpflow.defer_build():
        k1 = gpflow.kernels.RBF(2, lengthscales=5.0)
        k1.num_gauss_hermite_points = 5


    k1.compile()

    sess = k1.enquire_session()

