# Copyright 2017 Artem Artemev @awav, Hugh Salimbeni @hughsalimbeni
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

import functions
import tensorflow as tf
import numpy as np

from .. import settings


class NaturalGradient:
    """
    """
    def __init__(self, gamma=1.0, transform=None):
        self.gamma = gamma
        self.transform = transform
    

"""
This module implements the Gaussian parameter conversions between the following
three: (1) mean/vars, (2) natural parameters \theta or (3) expectation parameters \eta.
see https://twitter.com/HSalimbeni/status/875623985794367488
or ./docs/gaussian_conversions.jpeg
"""

def jittered_cholesky(M, jitter=None):
    """
    Add jitter and take Cholesky

    :param M: Tensor of shape ...xNxN
    :returns: The Cholesky decomposition of the input `M`.
        It's a `tf.Tensor` of shape ...xNxN
    """
    if jitter is None:
        jitter = settings.jitter 
    N = tf.shape(M)[-1]
    return tf.cholesky(M + jitter * tf.eye(N, dtype=M.dtype))


# TODO: requires generalization.
def forward_gradients(ys, xs, init_dxs):
    """
    Forward-mode push forward analogous to the pullback defined by `tf.gradients`.
    With `tf.gradients`, grad_ys is the vector being pulled back, and here d_xs is
    the vector being pushed forward.

    i.e. this computes (d ys / d xs)^T d_xs

    Modified from https://github.com/renmengye/tensorflow-forward-ad/issues/2.
    See https://j-towns.github.io/2017/06/12/A-new-trick.html for explanation
    of how this works.

    :param ys: Tensor variable being differentiated (wrt xs).
    :param xs: Tensor variable to differentiate wrt.
    :param d_xs: Tensor gradient to push forward.
    :return: The specified moment of the variational distribution.
    """
    with tf.name_scope('forward_gradients'):
        v = np.zeros(ys.shape)
        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=init_dxs)


def inverse_triangular(M):
    """
    Take inverse of lower triangular (e.g. Cholesky) matrix. This function
    broadcasts over the first index.

    :param M: Tensor with lower triangular structure of shape DxNxN.
    :returns: The inverse of the Cholesky decomposition.
        Output shape is same as intput.
    """
    N = tf.shape(M)[1]
    D = tf.shape(M)[0]

    dtype = M.dtype
    I_DNN = tf.eye(N, dtype=dtype)[None, :, :] * tf.ones((D, 1, 1), dtype=dtype)
    return tf.matrix_triangular_solve(M, I_DNN)


# The following functions expect their first and second inputs to have shape
# DN1 and DNN, respectively. Return values are also of shapes DN1 and DNN.


@swap_dimensions
def natural_to_meanvarsqrt(nat_1, nat_2):
    _natural_to_meanvarsqrt(nat_1, nat_2)


@swap_dimensions
def meanvarsqrt_to_expectation(m, v_sqrt):
    _meanvarsqrt_to_expectation(m, v_sqrt)


@swap_dimensions
def meanvarsqrt_to_natural(m, S_sqrt):
    _meanvarsqrt_to_natural(m, S_sqrt)


@swap_dimensions
def natural_to_expectation(nat_1, nat_2):
    mu, var_sqrt = _natural_to_meanvarsqrt(nat_1, nat_2)
    return _meanvarsqrt_to_expectation(mu, var_sqrt)


@swap_dimensions
def expectation_to_natural(eta_1, eta_2):
    nat_1, nat_2 = _expectation_to_meanvarsqrt(eta_1, eta_2)
    return _meanvarsqrt_to_natural(nat_1, nat_2)


@swap_dimensions
def expectation_to_meanvarsqrt(eta_1, eta_2):
    _expectation_to_meanvarsqrt(eta_1, eta_2)


def _expectation_to_meanvarsqrt(eta_1, eta_2):
    var = eta_2 - tf.matmul(eta_1, eta_1, transpose_b=True)
    return eta_1, jittered_cholesky(var)


def _meanvarsqrt_to_expectation(m, v_sqrt):
    v = tf.matmul(v_sqrt, v_sqrt, transpose_b=True)
    return m, v + tf.matmul(m, m, transpose_b=True)


def _natural_to_meanvarsqrt(nat_1, nat_2):
    var_sqrt_inv = tf.cholesky(-2 * nat_2)
    var_sqrt = inverse_triangular(var_sqrt_inv)
    S = tf.matmul(var_sqrt, var_sqrt, transpose_a=True)
    mu = tf.matmul(S, nat_1)
    # we want the decomposition of S as L L^T, not as L^T L, hence we need another cholesky
    return mu, jittered_cholesky(S)


def _meanvarsqrt_to_natural(m, S_sqrt):
    S_sqrt_inv = inverse_triangular(S_sqrt)
    S_inv = tf.matmul(S_sqrt_inv, S_sqrt_inv, transpose_a=True)
    return tf.matmul(S_inv, m), -0.5 * S_inv

def swap_dimensions(method):
    """
    Converts between GPflow indexing and tensorflow indexing
    `method` is a function that broadcasts over the first dimension
    (i.e. like all tensorflow matrix ops):
    * `method` inputs DN1, DNN
    * `method` outputs DN1, DNN

    returns a function that broadcasts over the final dimension (i.e. compatible with GPflow):
    inputs: ND, NND
    outputs: ND, NND
    """

    @functools.wraps(method)
    def wrapper(a_nd, b_nnd):
        with tf.name_scope('swap_dimensions'):
            a_dn1 = tf.transpose(a_nd)[:, :, None]
            b_dnn = tf.transpose(b_nnd, [2, 0, 1])
            with tf.name_scope(method.__name__):
                A_dn1, B_dnn = method(a_dn1, b_dnn)
            A_nd = tf.transpose(A_dn1[:, :, 0])
            B_nnd = tf.transpose(B_dnn, [1, 2, 0])
            return A_nd, B_nnd
    return wrapper