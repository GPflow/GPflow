# Copyright 2018 Hugh Salim, Artem Artemev @awav
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

import abc
import functools

import tensorflow as tf

from . import optimizer
from .. import settings
from ..models import Model


class NatGradOptimizer(optimizer.Optimizer):
    def __init__(self, gamma, **kwargs):
        super().__init__(**kwargs)
        self.name = self.__class__.__name__
        self._gamma = gamma
    
    @property
    def gamma(self):
        return self.gamma

    def minimize(self, model, session=None, var_list=None, feed_dict=None,
                 maxiter=1000, anchor=True, **kwargs):
        """
        Minimizes objective function of the model.

        :param model: GPflow model with objective tensor.
        :param session: Session where optimization will be run.
        :param var_list: List of extra variables which should be trained during optimization.
        :param feed_dict: Feed dictionary of tensors passed to session run method.
        :param maxiter: Number of run interation. Default value: 1000.
        :param anchor: If `True` trained variable values computed during optimization at
            particular session will be synchronized with internal parameter values.
        :param kwargs: This is a dictionary of extra parameters for session run method.
        """

        if model is None or not isinstance(model, Model):
            raise ValueError('Unknown type passed for optimization.')

        session = model.enquire_session(session)

        self._model = model

        with session.graph.as_default(), tf.name_scope(self.name):
            # Create optimizer variables before initialization.
            self._natgrad_op = self._build_natgrad_step_ops(*var_list)
            feed_dict = self._gen_feed_dict(model, feed_dict)
            for _i in range(maxiter):
                session.run(self._natgrad_op, feed_dict=feed_dict)

        if anchor:
            model.anchor(session)

    @staticmethod
    def _forward_gradients(ys, xs, d_xs):
        """
        Forward-mode pushforward analogous to the pullback defined by tf.gradients.
        With tf.gradients, grad_ys is the vector being pulled back, and here d_xs is
        the vector being pushed forward, i.e. this computes (d ys / d xs)^T d_xs.

        :param ys: list of variables being differentiated (tensor)
        :param xs: list of variables to differentiate wrt (tensor)
        :param d_xs: list of gradients to push forward (same shapes as ys)
        :return: the specified moment of the variational distribution
        """
        v = [tf.placeholder(y.dtype) for y in ys]
        g = tf.gradients(ys, xs, grad_ys=v)
        return tf.gradients(g, v, grad_ys=d_xs)

    def _build_natgrad_step_ops(self, *args):
        ops = []
        for arg in args:
           q_mu, q_sqrt = arg[:2]
           xi_transform = arg[2] if len(arg) > 2 else XiNat()
           ops.append(self._build_natgrad_step_op(q_mu, q_sqrt, xi_transform))
        ops = list(sum(ops, ()))
        return tf.group(ops)

    def _build_natgrad_step_op(self, q_mu_param, q_sqrt_param, xi_transform):
        """
        """
        objective = self._model.objective
        q_mu, q_sqrt = q_mu_param.constrained_tensor, q_sqrt_param.constrained_tensor

        etas = meanvarsqrt_to_expectation(q_mu, q_sqrt)
        nats = meanvarsqrt_to_natural(q_mu, q_sqrt)

        dL_d_mean, dL_d_varsqrt = tf.gradients(objective, [q_mu, q_sqrt])
        _nats = expectation_to_meanvarsqrt(*etas)
        dL_detas = tf.gradients(_nats, etas, grad_ys=[dL_d_mean, dL_d_varsqrt])

        _xis = xi_transform.naturals_to_xi(*nats)
        nat_dL_xis = self._forward_gradients(_xis, nats, dL_detas)

        xis = xi_transform.meanvarsqrt_to_xi(q_mu, q_sqrt)

        xis_new = [xi - self.gamma * nat_dL_xi for xi, nat_dL_xi in zip(xis, nat_dL_xis)]
        mean_new, varsqrt_new = xi_transform.xi_to_meanvarsqrt(*xis_new)
        mean_new.set_shape(q_mu_param.shape)
        varsqrt_new.set_shape(q_sqrt_param.shape)

        q_mu_u = q_mu_param.unconstrained_tensor
        q_sqrt_u = q_sqrt_param.unconstrained_tensor
        q_mu_assign = tf.assign(q_mu_u, q_mu_param.transform.backward_tensor(mean_new))
        q_sqrt_assign = tf.assign(q_sqrt_u, q_sqrt_param.transform.backward_tensor(varsqrt_new))
        return q_mu_assign, q_sqrt_assign

#
# Xi transformations necessary for natural gradient optimizer.
# Abstract class and two implementations: XiNat and XiSqrtMeanVar.
#

class XiTransform(metaclass=abc.ABCMeta):
    """
    XiTransform is the base class that implements three transformations necessary
    for the natural gradient calculation wrt any parameterization.
    This class does not handle any shape information, but it is assumed that
    the parameters pairs are always of shape (N, D) and (N, N, D).
    """
    @abc.abstractmethod
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        """
        Transforms the parameter `mean` and `varsqrt` to `xi_1`, `xi_2`

        :param mean: the mean parameter (N, D)
        :param varsqrt: the varsqrt parameter (N, N, D)
        :return: tuple (xi_1, xi_2), the xi parameters (N, D), (N, N, D)
        """
        pass

    @abc.abstractmethod
    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        """
        Transforms the parameter `xi_1`, `xi_2` to `mean`, `varsqrt`

        :param xi_1: the xi_1 parameter
        :param xi_2: the xi_2 parameter
        :return: tuple (mean, varsqrt), the meanvarsqrt parameters
        """
        pass

    @abc.abstractmethod
    def naturals_to_xi(self, nat_1, nat_2):
        """
        Applies the transform so that `nat_1`, `nat_2` is mapped to `xi_1`, `xi_2`

        :param nat_1: the nat_1 parameter
        :param nat_2: the nat_1 parameter
        :return: tuple `xi_1`, `xi_2`
        """
        pass


class XiNat(XiTransform):
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return meanvarsqrt_to_natural(mean, varsqrt)

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return natural_to_meanvarsqrt(xi_1, xi_2)

    def naturals_to_xi(self, nat_1, nat_2):
        return nat_1, nat_2


class XiSqrtMeanVar(XiTransform):
    def meanvarsqrt_to_xi(self, mean, varsqrt):
        return mean, varsqrt

    def xi_to_meanvarsqrt(self, xi_1, xi_2):
        return xi_1, xi_2

    def naturals_to_xi(self, nat_1, nat_2):
        return natural_to_meanvarsqrt(nat_1, nat_2)


#
# Auxiliary gaussian parameter conversion functions.
# 
# The following functions expect their first and second inputs to have shape
# DN1 and DNN, respectively. Return values are also of shapes DN1 and DNN.


def swap_dimensions(method):
    """
    Converts between GPflow indexing and tensorflow indexing
    `method` is a function that broadcasts over the first dimension (i.e. like all tensorflow matrix ops):
        `method` inputs DN1, DNN
        `method` outputs DN1, DNN
    :return: Function that broadcasts over the final dimension (i.e. compatible with GPflow):
        inputs: ND, NND
        outputs: ND, NND
    """
    @functools.wraps(method)
    def wrapper(a_nd, b_dnn, swap=True):
        if a_nd.get_shape().ndims != 2:
            raise ValueError("The `a_nd` input must have 2 dimentions.")
        if b_dnn.get_shape().ndims == 3:
            raise ValueError("The `b_nd` input must have 3 dimentions.")
        if swap:
            a_dn1 = tf.transpose(a_nd)[:, :, None]
            A_dn1, B_dnn = method(a_dn1, b_dnn)
            A_nd = tf.transpose(A_dn1[:, :, 0])
            return A_nd, B_dnn
        else:
            return method(a_dn1, b_dnn)
    return wrapper


@swap_dimensions
def natural_to_meanvarsqrt(nat_1, nat_2):
    var_sqrt_inv = tf.cholesky(-2 * nat_2)
    var_sqrt = _inverse_lower_triangular(var_sqrt_inv)
    s = tf.matmul(var_sqrt, var_sqrt, transpose_a=True)
    mu = tf.matmul(s, nat_1)
    # We need the decomposition of S as L L^T, not as L^T L,
    # hence we need another cholesky.
    return mu, _cholesky_with_jitter(s)


@swap_dimensions
def meanvarsqrt_to_natural(mu, s_sqrt):
    s_sqrt_inv = _inverse_lower_triangular(s_sqrt)
    s_inv = tf.matmul(s_sqrt_inv, s_sqrt_inv, transpose_a=True)
    return tf.matmul(s_inv, mu), -0.5 * s_inv


@swap_dimensions
def natural_to_expectation(nat_1, nat_2):
    return meanvarsqrt_to_expectation(*natural_to_meanvarsqrt(nat_1, nat_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_natural(eta_1, eta_2):
    return meanvarsqrt_to_natural(*expectation_to_meanvarsqrt(eta_1, eta_2, swap=False), swap=False)


@swap_dimensions
def expectation_to_meanvarsqrt(eta_1, eta_2):
    var = eta_2 - tf.matmul(eta_1, eta_1, transpose_b=True)
    return eta_1, _cholesky_with_jitter(var)


@swap_dimensions
def meanvarsqrt_to_expectation(m, v_sqrt):
    v = tf.matmul(v_sqrt, v_sqrt, transpose_b=True)
    return m, v + tf.matmul(m, m, transpose_b=True)


def _cholesky_with_jitter(M):
    """
    Add jitter and take Cholesky

    :param M: Tensor of shape NxNx...N
    :return: The Cholesky decomposition of the input `M`. It's a `tf.Tensor` of shape ...xNxN
    """
    N = tf.shape(M)[-1]
    return tf.cholesky(M + settings.jitter * tf.eye(N, dtype=N.dtype))

def _inverse_lower_triangular(M):
    """
    Take inverse of lower triangular (e.g. Cholesky) matrix. This function
    broadcasts over the first index.

    :param M: Tensor with lower triangular structure of shape DxNxN
    :return: The inverse of the Cholesky decomposition. Same shape as input.
    """
    if M.get_shape().ndims != 3:
        raise ValueError("Number of dimensions for input is required to be 3.")
    D, N, _ = tf.shape(M)
    I_DNN = tf.eye(N, dtype=N.dtype)[None, :, :] * tf.ones((D, 1, 1), dtype=M.dtype)
    return tf.matrix_triangular_solve(M, I_DNN)
