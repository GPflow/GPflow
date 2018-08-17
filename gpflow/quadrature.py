# Copyright 2017-2018 the GPflow authors.
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

import itertools
from collections import Iterable

import numpy as np
import tensorflow as tf

from . import settings
from .core.errors import GPflowError


def hermgauss(n: int):
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x.astype(settings.float_type), w.astype(settings.float_type)
    return x, w


def mvhermgauss(H: int, D: int):
    """
    Return the evaluation locations 'xn', and weights 'wn' for a multivariate
    Gauss-Hermite quadrature.

    The outputs can be used to approximate the following type of integral:
    int exp(-x)*f(x) dx ~ sum_i w[i,:]*f(x[i,:])

    :param H: Number of Gauss-Hermite evaluation points.
    :param D: Number of input dimensions. Needs to be known at call-time.
    :return: eval_locations 'x' (H**DxD), weights 'w' (H**D)
    """
    gh_x, gh_w = hermgauss(H)
    x = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


def mvnquad(func, means, covs, H: int, Din: int=None, Dout=None):
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.
    :param f: integrand function. Takes one input of shape ?xD.
    :param means: NxD
    :param covs: NxDxD
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
    to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures (N,*Dout)
    """
    # Figure out input shape information
    if Din is None:
        Din = means.shape[1] if type(means.shape) is tuple else means.shape[1].value

    if Din is None:
        raise GPflowError("If `Din` is passed as `None`, `means` must have a known shape. "
                          "Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
                          "is problematic. Consider using your own session.")  # pragma: no cover

    xn, wn = mvhermgauss(H, Din)
    N = tf.shape(means)[0]

    # transform points based on Gaussian parameters
    cholXcov = tf.cholesky(covs)  # NxDxD
    Xt = tf.matmul(cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True)  # NxDxH**D
    X = 2.0 ** 0.5 * Xt + tf.expand_dims(means, 2)  # NxDxH**D
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, Din))  # (H**D*N)xD

    # perform quadrature
    fevals = func(Xr)
    if Dout is None:
        Dout = tuple((d if type(d) is int else d.value) for d in fevals.shape[1:])

    if any([d is None for d in Dout]):
        raise GPflowError("If `Dout` is passed as `None`, the output of `func` must have known "
                          "shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
                          "is problematic. Consider using your own session.")  # pragma: no cover
    fX = tf.reshape(fevals, (H ** Din, N,) + Dout)
    wr = np.reshape(wn * np.pi ** (-Din * 0.5),
                    (-1,) + (1,) * (1 + len(Dout)))
    return tf.reduce_sum(fX * wr, 0)


def ndiagquad(funcs, H: int, Fmu, Fvar, logspace: bool=False, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Gauss-Hermite quadrature. The Gaussians must be independent.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise, on the following arguments:
        - `Din` positional arguments to match Fmu and Fvar; i.e., 1 if Fmu and Fvar are tensors;
          otherwise len(Fmu) (== len(Fvar)) positional arguments F1, F2, ...
        - the same keyword arguments as given by **Ys
        All arguments will be tensors of shape (N, 1)

    :param H: number of Gauss-Hermite quadrature points
    :param Fmu: array/tensor or `Din`-tuple/list thereof
    :param Fvar: array/tensor or `Din`-tuple/list thereof
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N` (i.e., shape (N,) or (N, 1))
    :return: shape is the same as that of the first Fmu


    Example use-cases:

        Fmu, Fvar are mean and variance of the latent GP, can be shape (N, 1) or (N,)
        m1, m2 are 'scalar' functions of a single argument F, broadcasting over arrays

        Em1, Em2 = ndiagquad([m1, m2], 50, Fmu, Fvar)
            calculates Em1 = ∫ m1(F) N(F; Fmu, Fvar) dF and Em2 = ∫ m2(F) N(F; Fmu, Fvar) dF
            for each of the elements of Fmu and Fvar. Em1 and Em2 have the same shape as Fmu.

        logp is a 'scalar' function of F and Y
        Y are the observations, with shape (N,) or (N, 1) with same length as Fmu and Fvar

        Ev = ndiagquad(logp, 50, Fmu, Fvar, Y=Y)
            calculates Ev = ∫ logp(F, Y) N(F; Fmu, Fvar) dF (variational expectations)
            for each of the elements of Y, Fmu and Fvar. Ev has the same shape as Fmu.

        Ep = ndiagquad(logp, 50, Fmu, Fvar, logspace=True, Y=Y)
            calculates Ep = log ∫ exp(logp(F, Y)) N(F; Fmu, Fvar) dF (predictive density)
            for each of the elements of Y, Fmu and Fvar. Ep has the same shape as Fmu.

        Heteroskedastic likelihoods:
        g1, g2 are now functions of both F and G
        logp is a function of F, G and Y
        Gmu, Gvar are mean and variance of a different GP controlling the variance

        Em = ndiagquad(m1, 50, Fmu, Fvar)
            -> Em1 = ∫∫ m1(F, G) N(F; Fmu, Fvar) N(G; Gmu, Gvar) dF dG

        Ev = ndiagquad(logp, 50, Fmu, Fvar, Y=Y)
            -> Ev = ∫∫ logp(F, G, Y) N(F; Fmu, Fvar) N(G; Gmu, Gvar) dF dG
               (variational expectations)

        Ep = ndiagquad(logp, 50, Fmu, Fvar, logspace=True, Y=Y)
            -> Ep = log ∫∫ exp(logp(F, G, Y)) N(F; Fmu, Fvar) N(G; Gmu, Gvar) dF dG
               (predictive density)
    """
    def unify(f_list):
        """
        Stack a list of means/vars into a full block
        """
        return tf.reshape(
                tf.concat([tf.reshape(f, (-1, 1)) for f in f_list], axis=1),
                (-1, 1, Din))

    if isinstance(Fmu, (tuple, list)):
        Din = len(Fmu)
        shape = tf.shape(Fmu[0])
        Fmu, Fvar = map(unify, [Fmu, Fvar])    # both N x 1 x Din
    else:
        Din = 1
        shape = tf.shape(Fmu)
        Fmu, Fvar = [tf.reshape(f, (-1, 1, 1)) for f in [Fmu, Fvar]]

    xn, wn = mvhermgauss(H, Din)
    # xn: H**Din x Din, wn: H**Din

    gh_x = xn.reshape(1, -1, Din)             # 1 x H**Din x Din
    Xall = gh_x * tf.sqrt(2.0 * Fvar) + Fmu   # N x H**Din x Din
    Xs = [Xall[:, :, i] for i in range(Din)]  # N x H**Din  each

    gh_w = wn * np.pi ** (-0.5 * Din)  # H**Din x 1

    for name, Y in Ys.items():
        Y = tf.reshape(Y, (-1, 1))
        Y = tf.tile(Y, [1, H**Din])  # broadcast Y to match X
        # without the tiling, some calls such as tf.where() (in bernoulli) fail
        Ys[name] = Y  # now N x H**Din

    def eval_func(f):
        feval = f(*Xs, **Ys)  # f should be elementwise: return shape N x H**Din
        if logspace:
            log_gh_w = np.log(gh_w.reshape(1, -1))
            result = tf.reduce_logsumexp(feval + log_gh_w, axis=1)
        else:
            result = tf.matmul(feval, gh_w.reshape(-1, 1))
        return tf.reshape(result, shape)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)


def ndiag_mc(funcs, S: int, Fmu, Fvar, logspace: bool=False, epsilon=None, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name

    Fmu, Fvar, Ys should all have same shape, with overall size `N`
    :return: shape is the same as that of the first Fmu
    """
    N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1]

    if epsilon is None:
        epsilon = tf.random_normal((S, N, D), dtype=settings.float_type)

    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))

    for name, Y in Ys.items():
        D_out = tf.shape(Y)[1]
        # we can't rely on broadcasting and need tiling
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # S x N x D_out
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # S * N x D_out

    def eval_func(func):
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.log(tf.cast(S, settings.float_type))
            return tf.reduce_logsumexp(feval, axis=0) - log_S  # N x D
        else:
            return tf.reduce_mean(feval, axis=0)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)
