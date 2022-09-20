# Copyright 2017-2020 The GPflow Contributors. All Rights Reserved.
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
import warnings
from functools import wraps
from typing import Callable, Iterable, List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf

from ..base import AnyNDArray, TensorType
from ..config import default_float
from ..experimental.check_shapes import check_shapes
from ..utilities import to_default_float
from .gauss_hermite import NDiagGHQuadrature


@check_shapes(
    "return[0]: [n_quad_points]",
    "return[1]: [n_quad_points]",
)
def hermgauss(n: int) -> Tuple[AnyNDArray, AnyNDArray]:
    # Type-ignore is because for some versions mypy can't find np.polynomial.hermite
    x, w = np.polynomial.hermite.hermgauss(n)
    x, w = x.astype(default_float()), w.astype(default_float())
    return x, w


@check_shapes(
    "return[0]: [n_quad_points, D]",
    "return[1]: [n_quad_points]",
)
def mvhermgauss(H: int, D: int) -> Tuple[AnyNDArray, AnyNDArray]:
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
    x: AnyNDArray = np.array(list(itertools.product(*(gh_x,) * D)))  # H**DxD
    w = np.prod(np.array(list(itertools.product(*(gh_w,) * D))), 1)  # H**D
    return x, w


@check_shapes(
    "means: [N, Din]",
    "covs: [N, Din, Din]",
    "return: [N, Dout...]",
)
def mvnquad(
    func: Callable[[tf.Tensor], tf.Tensor],
    means: TensorType,
    covs: TensorType,
    H: int,
    Din: Optional[int] = None,
    Dout: Optional[Tuple[int, ...]] = None,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of a single function 'f'
    using Gauss-Hermite quadrature.

    :param f: integrand function. Takes one input of shape ?xD.
    :param H: Number of Gauss-Hermite evaluation points.
    :param Din: Number of input dimensions. Needs to be known at call-time.
    :param Dout: Number of output dimensions. Defaults to (). Dout is assumed
        to leave out the item index, i.e. f actually maps (?xD)->(?x*Dout).
    :return: quadratures
    """
    # Figure out input shape information
    if Din is None:
        Din = means.shape[1]

    if Din is None:
        raise ValueError(
            "If `Din` is passed as `None`, `means` must have a known shape. "
            "Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
            "is problematic. Consider using your own session."
        )  # pragma: no cover

    xn, wn = mvhermgauss(H, Din)
    N = means.shape[0]

    # transform points based on Gaussian parameters
    cholXcov = tf.linalg.cholesky(covs)  # NxDxD
    Xt = tf.linalg.matmul(
        cholXcov, tf.tile(xn[None, :, :], (N, 1, 1)), transpose_b=True
    )  # NxDxH**D
    X = 2.0 ** 0.5 * Xt + tf.expand_dims(means, 2)  # NxDxH**D
    Xr = tf.reshape(tf.transpose(X, [2, 0, 1]), (-1, Din))  # (H**D*N)xD

    # perform quadrature
    fevals = func(Xr)
    if Dout is None:
        Dout = tuple((d if type(d) is int else d.value) for d in fevals.shape[1:])

    if any([d is None for d in Dout]):
        raise ValueError(
            "If `Dout` is passed as `None`, the output of `func` must have known "
            "shape. Running mvnquad in `autoflow` without specifying `Din` and `Dout` "
            "is problematic. Consider using your own session."
        )  # pragma: no cover
    fX = tf.reshape(
        fevals,
        (
            H ** Din,
            N,
        )
        + Dout,
    )
    wr = np.reshape(wn * np.pi ** (-Din * 0.5), (-1,) + (1,) * (1 + len(Dout)))
    return tf.reduce_sum(fX * wr, 0)


@check_shapes(
    "Fmu: [broadcast Din, N...]",
    "Fvar: [broadcast Din, N...]",
    "Ys.values(): [N...]",
    "return: [broadcast Dout, N...]",
)
def ndiagquad(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    H: int,
    Fmu: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    Fvar: Union[TensorType, Tuple[TensorType, ...], List[TensorType]],
    logspace: bool = False,
    **Ys: TensorType,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Gauss-Hermite quadrature. The Gaussians must be independent.

    The means and variances of the Gaussians are specified by Fmu and Fvar.
    The N-integrals are assumed to be taken wrt the last dimensions of Fmu, Fvar.

    `Fmu`, `Fvar`, `Ys` should all have same shape, with overall size `N`.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param H: number of Gauss-Hermite quadrature points
    :param Fmu: array/tensor or `Din`-tuple/list thereof
    :param Fvar: array/tensor or `Din`-tuple/list thereof
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param Ys: arrays/tensors; deterministic arguments to be passed by name
    :return: shape is the same as that of the first Fmu
    """
    warnings.warn(
        "Please use gpflow.quadrature.NDiagGHQuadrature instead "
        "(note the changed convention of how multi-dimensional quadrature is handled)",
        DeprecationWarning,
    )

    n_gh = H
    if isinstance(Fmu, (tuple, list)):
        dim = len(Fmu)
        shape = tf.shape(Fmu[0])
        Fmu = tf.stack(Fmu, axis=-1)
        Fvar = tf.stack(Fvar, axis=-1)
    else:
        dim = 1
        shape = tf.shape(Fmu)

    Fmu = tf.reshape(Fmu, (-1, dim))
    Fvar = tf.reshape(Fvar, (-1, dim))

    Ys = {Yname: tf.reshape(Y, (-1, 1)) for Yname, Y in Ys.items()}

    def wrapper(old_fun: Callable[..., tf.Tensor]) -> Callable[..., tf.Tensor]:
        @wraps(old_fun)
        def new_fun(X: TensorType, **Ys: TensorType) -> tf.Tensor:
            Xs = tf.unstack(tf.expand_dims(X, axis=-2), axis=-1)
            fun_eval = old_fun(*Xs, **Ys)
            return tf.cond(
                pred=tf.less(tf.rank(fun_eval), tf.rank(X)),
                true_fn=lambda: fun_eval[..., tf.newaxis],
                false_fn=lambda: fun_eval,
            )

        return new_fun

    if isinstance(funcs, Iterable):
        funcs = [wrapper(f) for f in funcs]
    else:
        funcs = wrapper(funcs)

    quadrature = NDiagGHQuadrature(dim, n_gh)
    if logspace:
        result = quadrature.logspace(funcs, Fmu, Fvar, **Ys)
    else:
        result = quadrature(funcs, Fmu, Fvar, **Ys)

    if isinstance(result, list):
        result = [tf.reshape(r, shape) for r in result]
    else:
        result = tf.reshape(result, shape)

    return result


@check_shapes(
    "Fmu: [N, Din]",
    "Fvar: [N, Din]",
    "Ys.values(): [broadcast N, .]",
    "return: [broadcast n_funs, N, P]",
)
def ndiag_mc(
    funcs: Union[Callable[..., tf.Tensor], Iterable[Callable[..., tf.Tensor]]],
    S: int,
    Fmu: TensorType,
    Fvar: TensorType,
    logspace: bool = False,
    epsilon: Optional[TensorType] = None,
    **Ys: TensorType,
) -> tf.Tensor:
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Monte Carlo samples. The Gaussians must be independent.

    `Fmu`, `Fvar`, `Ys` should all have same shape, with overall size `N`.

    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param S: number of Monte Carlo sampling points
    :param Fmu: array/tensor
    :param Fvar: array/tensor
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param Ys: arrays/tensors; deterministic arguments to be passed by name
    :return: shape is the same as that of the first Fmu
    """
    N, D = tf.shape(Fmu)[0], tf.shape(Fvar)[1]

    if epsilon is None:
        epsilon = tf.random.normal(shape=[S, N, D], dtype=default_float())

    mc_x = Fmu[None, :, :] + tf.sqrt(Fvar[None, :, :]) * epsilon
    mc_Xr = tf.reshape(mc_x, (S * N, D))

    for name, Y in Ys.items():
        D_out = Y.shape[1]
        # we can't rely on broadcasting and need tiling
        mc_Yr = tf.tile(Y[None, ...], [S, 1, 1])  # [S, N, D]_out
        Ys[name] = tf.reshape(mc_Yr, (S * N, D_out))  # S * [N, _]out

    def eval_func(func: Callable[..., tf.Tensor]) -> tf.Tensor:
        feval = func(mc_Xr, **Ys)
        feval = tf.reshape(feval, (S, N, -1))
        if logspace:
            log_S = tf.math.log(to_default_float(S))
            return tf.reduce_logsumexp(feval, axis=0) - log_S  # [N, D]
        else:
            return tf.reduce_mean(feval, axis=0)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]
    else:
        return eval_func(funcs)
