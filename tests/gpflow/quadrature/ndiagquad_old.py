from collections.abc import Iterable

import numpy as np
import tensorflow as tf

from gpflow.quadrature.deprecated import mvhermgauss


def ndiagquad(funcs, H: int, Fmu, Fvar, logspace: bool = False, **Ys):
    """
    Computes N Gaussian expectation integrals of one or more functions
    using Gauss-Hermite quadrature. The Gaussians must be independent.
    The means and variances of the Gaussians are specified by Fmu and Fvar.
    The N-integrals are assumed to be taken wrt the last dimensions of Fmu, Fvar.
    :param funcs: the integrand(s):
        Callable or Iterable of Callables that operates elementwise
    :param H: number of Gauss-Hermite quadrature points
    :param Fmu: array/tensor or `Din`-tuple/list thereof
    :param Fvar: array/tensor or `Din`-tuple/list thereof
    :param logspace: if True, funcs are the log-integrands and this calculates
        the log-expectation of exp(funcs)
    :param **Ys: arrays/tensors; deterministic arguments to be passed by name
    Fmu, Fvar, Ys should all have same shape, with overall size `N`
    :return: shape is the same as that of the first Fmu
    """
    if isinstance(Fmu, (tuple, list)):
        Din = len(Fmu)

        def unify(f_list):
            """Stack a list of means/vars into a full block."""
            return tf.reshape(
                tensor=tf.concat([tf.reshape(f, shape=(-1, 1)) for f in f_list], axis=1),
                shape=(-1, 1, Din),
            )

        shape = tf.shape(Fmu[0])
        Fmu, Fvar = map(unify, [Fmu, Fvar])  # both [N, 1, Din]

        print(Fmu)
        print(Fvar)
    else:
        Din = 1
        shape = tf.shape(Fmu)
        Fmu, Fvar = [tf.reshape(f, (-1, 1, 1)) for f in [Fmu, Fvar]]

    xn, wn = mvhermgauss(H, Din)
    # xn: H**Din x Din, wn: H**Din

    gh_x = xn.reshape(1, -1, Din)  # [1, H]**Din x Din
    Xall = gh_x * tf.sqrt(2.0 * Fvar) + Fmu  # [N, H]**Din x Din
    Xs = [Xall[:, :, i] for i in range(Din)]  # [N, H]**Din  each

    gh_w = wn * np.pi ** (-0.5 * Din)  # H**Din x 1

    for name, Y in Ys.items():
        Y = tf.reshape(Y, (-1, 1))
        Y = tf.tile(Y, [1, H ** Din])  # broadcast Y to match X
        # without the tiling, some calls such as tf.where() (in bernoulli) fail
        Ys[name] = Y  # now [N, H]**Din

    def eval_func(f):
        feval = f(*Xs, **Ys)  # f should be elementwise: return shape [N, H]**Din
        if logspace:
            log_gh_w = np.log(gh_w.reshape(1, -1))
            result = tf.reduce_logsumexp(feval + log_gh_w, axis=1)
        else:
            result = tf.linalg.matmul(feval, gh_w.reshape(-1, 1))
        return tf.reshape(result, shape)

    if isinstance(funcs, Iterable):
        return [eval_func(f) for f in funcs]

    return eval_func(funcs)
